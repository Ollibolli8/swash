import * as THREE from "three/webgpu";
import { Fn, If, Loop, abs, clamp, cos, exp, float, floor, instanceIndex, int, ivec2, length, log, max, min, mix, select, sin, sqrt, storage, texture as sampleTexture, textureStore, uniform, vec2, vec4 } from "three/tsl";
import { noise } from "./light";

/**
 * The surf, simulated: the shallow-water equations on the GPU over a strip of beach, the way coastal engineers model
 * a swash zone, at the scale of the view (a few centimetres a cell, a few hundred steps a second).
 *
 * Waves enter from the sea already shoaled (steep-fronted, each with its own height along the shore), steepen into
 * bores over the shallows, run up the beach face, stop, and drain back down it into the next one. Water that runs
 * into a channel someone has drawn or written in the sand flows down it. Foam is made where a wave breaks (harder along some
 * stretches of a crest than others), its fresh mat bursts into lace within a second, and the lace is carried by the
 * actual flow (piled up at the running edge, torn apart where the water spreads), left on the sand when it drains,
 * and popped there within seconds; sand stirred up by the bores clouds the water and settles; the sand stays wet
 * behind the water and dries.
 *
 * Numerics: a staggered grid (depth at cell centres, velocities on faces); semi-Lagrangian momentum with a little
 * eddy viscosity; upwind, mass-conserving continuity with hydrostatic face depths, so the water stays put on a slope
 * and wet/dry fronts move without leaks; Manning bed friction; a relaxation zone at the seaward edge that feeds the
 * incoming waves and absorbs what the beach reflects. The strip is periodic along the shore, so any screen width
 * tiles it seamlessly. Units are metres and seconds. After Chentanez & Müller (2010) and the textbook staggered
 * shallow-water scheme (Bridson, "Fluid Simulation for Computer Graphics", 2nd ed.).
 */

type F = THREE.Node<"float">;
type V2 = THREE.Node<"vec2">;
type V4 = THREE.Node<"vec4">;

/** The simulated strip, metres: periodic along the shore (W), from Y0 on the sand to Y1 at sea. The still-water
 *  line lies OFF metres seaward of world y = 0. */
export const SURF = { W: 9.6, Y0: -2.4, Y1: 3.6, OFF: -0.2, SPONGE: 0.8 } as const;
const G = 9.81;
/** Lace reset period: the foam pattern is carried by the flow for this long, then handed to its twin. */
export const LACE_PERIOD = 1.4;

/** Hermite step from a to b; either order (b < a falls instead of rising). */
const smooth = (a: number, b: number, x: F) => { const t = clamp(x.sub(a).div(b - a), 0, 1); return t.mul(t).mul(t.mul(-2).add(3)); };

/** Smooth ramp: 0 below `at`, slope 1 above, rounded over `k`. */
const ramp = (x: F, at: number, k: number) => {
  const d = x.sub(at);
  return max(d, 0).add(log(exp(abs(d).div(-k)).add(1)).mul(k));
};

/** How fast a channel fills (per second) under water `wet` (0 to 1) running at `speed` m/s: slowly in still water,
 *  quickly under a running swash, which carries sand roughly as the square of its speed. */
const filling = (wet: F, speed: F) => wet.mul(speed.mul(speed).mul(0.6).add(0.03)).add(0.002);

/** A seeded, deterministic random sequence for the seabed's relief. */
const random = (() => { let s = 0x2f6b; return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); })();
const RELIEF = Array.from({ length: 6 }, (_, i) => ({ n: i + 1, ky: 0.6 + random() * 2.4, px: random() * 6.283, py: random() * 6.283 }));
const TAU = Math.PI * 2;

/**
 * The beach, metres above still water (negative under it), at metres (x, y): a 1:7 beach face with low cusps, a 1:4
 * terrace where the waves run in as bores, its outer edge uneven along the shore, a steeper step down and deep water
 * beyond (4.4 m), and a little relief everywhere, all periodic along the shore.
 */
export function bedAt(x: F, y: F): F {
  const s = y.sub(SURF.OFF).toVar();
  const depth = s.div(7).add(ramp(s, 0.3, 0.25).mul(1 / 4 - 1 / 7)).add(ramp(s, 1.6, 0.35).mul(1 / 1.8 - 1 / 4))
    .add(ramp(s, 3.5, 0.6).mul(1 / 2.5 - 1 / 1.8)).toVar();
  // Smoothly no deeper than 4.4 m.
  const capped = float(4.4).sub(ramp(float(4.4).sub(depth), 0, 0.4));
  const k = TAU / SURF.W;
  const cusps = cos(x.mul(3 * k).add(0.4)).mul(0.6).add(cos(x.mul(4 * k).add(2.1)).mul(0.3)).add(cos(x.mul(5 * k).add(4.4)).mul(0.25))
    .mul(0.025).mul(exp(s.add(0.5).div(0.9).pow(2).negate()));
  // The terrace's outer edge rises and falls along the shore (a low bar, a shallow channel through it): waves break
  // early over the bar and late in the channel, and their bores bend as they cross the uneven terrace.
  const bar = cos(x.mul(k).add(1.1)).mul(0.6).add(cos(x.mul(2 * k).add(3.0)).mul(0.4)).add(cos(x.mul(3 * k).add(5.2)).mul(0.25))
    .mul(0.055).mul(exp(s.sub(1.3).div(0.8).pow(2).negate()));
  let relief: F = float(0);
  for (const r of RELIEF) relief = relief.add(cos(x.mul(r.n * k).add(r.px)).mul(cos(s.mul(r.ky).add(r.py))).mul(0.012 / Math.sqrt(r.n)));
  return capped.negate().add(cusps).add(bar).add(relief.mul(clamp(s.add(0.5), 0, 1)));
}

/** Still-water depth at y (metres, no relief), for the speed of the incoming waves. */
const depthJS = (y: number) => {
  const sp = (x: number, at: number, k: number) => { const d = x - at; return Math.max(d, 0) + Math.log(Math.exp(-Math.abs(d) / k) + 1) * k; };
  const s = y - SURF.OFF;
  const d = s / 7 + sp(s, 0.3, 0.25) * (1 / 4 - 1 / 7) + sp(s, 1.6, 0.35) * (1 / 1.8 - 1 / 4) + sp(s, 3.5, 0.6) * (1 / 2.5 - 1 / 1.8);
  return 4.4 - sp(4.4 - d, 0, 0.4);
};

/**
 * The incoming waves, a Baltic wind sea on a fair afternoon (significant height about 0.25 m at the strip's edge):
 * three trains, each a steep-fronted wave (five harmonics of a sawtooth) whose height varies along its crest and
 * from one wave to the next. They run straight in: the strip is too narrow for their wavelength to fit at an angle.
 */
const NH = 5;
const TRAINS = [{ T: 4.2, a: 0.1, phase: 0.3, seed: 1.7 }, { T: 3.7, a: 0.06, phase: 2.1, seed: 5.3 }, { T: 5.0, a: 0.05, phase: 4.4, seed: 9.1 }]
  .map(t => { const w = TAU / t.T, c = Math.sqrt(G * depthJS(SURF.Y1)); return { ...t, w, k: w / c, c }; });
/** A hash of a wave's number, 0..1. */
const hashN = (n: F, seed: number) => {
  const p = n.mul(0.1031).add(seed).toVar();
  const f = p.sub(floor(p));
  const q = f.mul(f.add(33.33));
  return q.mul(q.add(q)).sub(floor(q.mul(q.add(q))));
};

export type Sea = { eta: F; slope: V2; flow: F };

/**
 * The incoming sea at metres (x, y) and time t: surface height, its slope, and the shoreward flow speed of a
 * long wave (eta c / depth). Seaward of the strip the harmonics fade with depth, so the swell rounds out offshore.
 */
export function incoming(x: F, y: F, t: F): Sea {
  const eta = float(0).toVar(), gx = float(0).toVar(), gy = float(0).toVar();
  const smooth = exp(max(y.sub(SURF.Y1), 0).div(-2.5)).toVar();
  const k = TAU / SURF.W;
  for (const train of TRAINS) {
    const phi = y.sub(SURF.Y1).mul(train.k).add(t.mul(train.w)).add(train.phase).toVar();
    // Each wave (one turn of the phase) gets its own heights along the crest.
    const n = floor(phi.div(TAU)).toVar();
    const along = cos(x.mul(k).add(hashN(n, train.seed).mul(TAU))).mul(0.5)
      .add(cos(x.mul(5 * k).add(hashN(n, train.seed + 1.3).mul(TAU))).mul(0.14))
      .add(cos(x.mul(7 * k).add(hashN(n, train.seed + 2.9).mul(TAU))).mul(0.1))
      .add(cos(x.mul(2 * k).add(hashN(n, train.seed + 3.1).mul(TAU))).mul(0.32))
      .add(cos(x.mul(3 * k).add(hashN(n, train.seed + 7.7).mul(TAU))).mul(0.22));
    const dAlong = sin(x.mul(k).add(hashN(n, train.seed).mul(TAU))).mul(-0.5 * k)
      .sub(sin(x.mul(5 * k).add(hashN(n, train.seed + 1.3).mul(TAU))).mul(0.14 * 5 * k))
      .sub(sin(x.mul(7 * k).add(hashN(n, train.seed + 2.9).mul(TAU))).mul(0.1 * 7 * k))
      .sub(sin(x.mul(2 * k).add(hashN(n, train.seed + 3.1).mul(TAU))).mul(0.32 * 2 * k))
      .sub(sin(x.mul(3 * k).add(hashN(n, train.seed + 7.7).mul(TAU))).mul(0.22 * 3 * k));
    const m = along.mul(0.38).add(1).toVar();
    const s = float(0).toVar(), ds = float(0).toVar();
    for (let h = 1; h <= NH; h++) {
      const x0 = h / (NH + 1), sigma = Math.sin(Math.PI * x0) / (Math.PI * x0);
      const fade = h === 1 ? 1 : smooth.pow(h - 1);
      s.addAssign(sin(phi.mul(h)).mul(fade).mul((2 / Math.PI) * sigma / h));
      ds.addAssign(cos(phi.mul(h)).mul(fade).mul((2 / Math.PI) * sigma));
    }
    eta.addAssign(s.mul(m).mul(train.a));
    gx.addAssign(s.mul(dAlong.mul(0.38)).mul(train.a));
    gy.addAssign(ds.mul(m).mul(train.a * train.k));
  }
  const still = max(bedAt(x, y).negate(), 0.15);
  const flow = eta.mul(Math.sqrt(G)).div(sqrt(still));
  return { eta, slope: vec2(gx, gy), flow };
}

export type Brush = THREE.UniformNode<"vec4", THREE.Vector4>;

/** Bands along the shore that the sound is measured in. */
export const BANDS = 8;

export type Surf = {
  /** (surface height, its slope x, slope y, water depth) */
  surface: THREE.StorageTexture;
  /** lace displacement, metres: (layer A xy, layer B xy) */
  drift: THREE.StorageTexture;
  /** (foam, stirred sand, wet sand, foam left on the sand) */
  matter: THREE.StorageTexture;
  /** The drawn channels, four times finer than the water's cells: (depth below the beach m, its slope x, slope y, 0);
   *  negative on the sand pushed up beside a channel. */
  groove: THREE.StorageTexture;
  nx: number; ny: number; dt: number;
  /** Strokes on the beach this frame, metres: (from xy, to xy); a stroke on water pushes it, one on sand digs. */
  brushes: Brush[];
  brushRadius: THREE.UniformNode<"float", number>;
  push: THREE.UniformNode<"vec2", THREE.Vector2>;
  /** Writing: a canvas over the whole strip, `inkScale` pixels a metre, whose green channel falls from 1 on a
   *  letter's stroke to 0 two radii from it (writing.ts). It is pressed into the dry sand left of `inkTo` (metres
   *  along the shore) while `inkOn` is 1, as a trench like a finger's. */
  ink: THREE.CanvasTexture;
  inkScale: number;
  inkOn: THREE.UniformNode<"float", number>;
  inkTo: THREE.UniformNode<"float", number>;
  /** The word's box and the last word's, metres (from xy, to xy): the hand smooths the sand in them flat just ahead
   *  of the writing. */
  inkBox: THREE.UniformNode<"vec4", THREE.Vector4>;
  inkWas: THREE.UniformNode<"vec4", THREE.Vector4>;
  /** Advance the water by `steps` fixed steps and refresh the textures the renderer reads. */
  run: (steps: number) => void;
  /** One more pass of the drawn grooves with no time passing. Their slopes are taken from the pass before, so a mark
   *  made while the sea stands still needs this to show its relief. */
  settle: () => void;
  /** What the surf sounds like now, in BANDS bands along the shore: (breaking bores, the swash running over the sand,
   *  foam bursting, the water's motion) for each band, each about 0 to 1. Resolves a frame or two later. */
  measure: () => Promise<Float32Array>;
  /** Seconds simulated since the start. */
  time: () => number;
  reset: (seconds?: number) => void;
  dispose: () => void;
};

/**
 * Builds the solver. `fine` picks 3.75 cm cells (256 × 160, 300 steps a second); otherwise 5 cm (192 × 120, 225).
 */
export function createSurf(renderer: THREE.WebGPURenderer, fine: boolean): Surf {
  const dx = fine ? 0.0375 : 0.05;
  const NX = Math.round(SURF.W / dx), NY = Math.round((SURF.Y1 - SURF.Y0) / dx), COUNT = NX * NY;
  const dt = dx / 11.25;
  const k = dt / dx;
  const UMAX = 0.25 / k;
  const EPS = 0.001;
  const cleanup: Array<() => void> = [];
  const buffer = () => { const b = new THREE.StorageBufferAttribute(COUNT, 4); return b; };
  // State (depth, u on the east face, v on the seaward face, channel depth); matter; lace displacement.
  const S = [buffer(), buffer()], E = [buffer(), buffer()], L = [buffer(), buffer()];
  const bedBuffer = new THREE.StorageBufferAttribute(COUNT, 1);
  const clockBuffer = new THREE.StorageBufferAttribute(new Float32Array([0, 0, 0, 0]), 4);
  const texture = () => {
    const t = new THREE.StorageTexture(NX, NY);
    t.type = THREE.HalfFloatType; t.format = THREE.RGBAFormat;
    t.wrapS = THREE.RepeatWrapping; t.wrapT = THREE.ClampToEdgeWrapping;
    t.minFilter = t.magFilter = THREE.LinearFilter; t.generateMipmaps = false;
    cleanup.push(() => t.dispose()); return t;
  };
  const surface = texture(), drift = texture(), matter = texture();

  const brushes = Array.from({ length: 4 }, () => uniform(new THREE.Vector4(-99, -99, -99, -99)));
  const brushRadius = uniform(0.06), push = uniform(new THREE.Vector2());
  const dig = uniform(0);
  const startTime = uniform(0);

  // The writing, on the channels' fine grid (four pixels a cell).
  const inkCanvas = document.createElement("canvas");
  inkCanvas.width = NX * 4; inkCanvas.height = NY * 4;
  const ink = new THREE.CanvasTexture(inkCanvas);
  ink.flipY = false; ink.colorSpace = THREE.NoColorSpace;
  ink.wrapS = THREE.RepeatWrapping; ink.wrapT = THREE.ClampToEdgeWrapping;
  ink.generateMipmaps = true; ink.minFilter = THREE.LinearMipmapLinearFilter; ink.magFilter = THREE.LinearFilter;
  cleanup.push(() => ink.dispose());
  const inkOn = uniform(0), inkTo = uniform(-99);
  const inkBox = uniform(new THREE.Vector4(-99, -99, -99, -99)), inkWas = uniform(new THREE.Vector4(-99, -99, -99, -99));
  const inside = (p: V2, box: THREE.UniformNode<"vec4", THREE.Vector4>) => smooth(-0.04, 0.02, p.x.sub(box.x)).mul(smooth(0.04, -0.02, p.x.sub(box.z)))
    .mul(smooth(-0.04, 0.02, p.y.sub(box.y))).mul(smooth(0.04, -0.02, p.y.sub(box.w)));
  /** 1 where the hand is smoothing the sand for the word (in its box or the last word's, behind the hand and just
   *  ahead of it), else 0. */
  const smoothed = (p: V2) => max(inside(p, inkBox), inside(p, inkWas)).mul(smooth(0.3, 0.1, p.x.sub(inkTo))).mul(inkOn);
  /** Distance from a letter's stroke at metres p, in stroke radii (2 and beyond: no letter near), at mip `level`. */
  const inked = (p: V2, level: number) => {
    const g = (sampleTexture(ink, vec2(p.x.div(SURF.W), p.y.sub(SURF.Y0).div(SURF.Y1 - SURF.Y0))).level(float(level)) as unknown as V4).y;
    // Written left to right: nothing beyond the hand yet.
    const written = smooth(0.02, -0.02, p.x.sub(inkTo)).mul(inkOn);
    return float(1).sub(g.mul(written)).mul(2);
  };

  const bed = storage(bedBuffer, "float", COUNT);
  const clock = storage(clockBuffer, "vec4", 1);

  // Grid helpers: x wraps (periodic shore), y clamps.
  const wrap = (i: F) => i.sub(floor(i.div(NX)).mul(NX));
  const clampJ = (j: F) => clamp(j, 0, NY - 1);
  const at = (i: F, j: F) => int(j.mul(NX).add(i));
  const cellX = (i: F) => i.add(0.5).mul(dx);
  const cellY = (j: F) => j.add(0.5).mul(dx).add(SURF.Y0);
  type Buf = THREE.StorageBufferNode<"vec4">;
  const load = (b: Buf, i: F, j: F) => b.element(at(i, j)) as unknown as V4;
  const bilerp = (b: Buf, fx: F, fy: F): V4 => {
    const x0 = floor(fx).toVar(), y0 = floor(fy).toVar();
    const tx = fx.sub(x0), ty = fy.sub(y0);
    const xa = wrap(x0), xb = wrap(x0.add(1)), ya = clampJ(y0), yb = clampJ(y0.add(1));
    return mix(mix(load(b, xa, ya), load(b, xb, ya), tx), mix(load(b, xa, yb), load(b, xb, yb), tx), ty);
  };
  const ij = () => {
    const i = float(instanceIndex.mod(NX)).toVar(), j = float(instanceIndex.div(NX)).toVar();
    return { i, j };
  };
  const segment = (p: V2, brush: Brush) => {
    const a = brush.xy, ab = brush.zw.sub(a);
    const s = clamp(p.sub(a).dot(ab).div(max(ab.dot(ab), 1e-8)), 0, 1);
    return length(p.sub(a.add(ab.mul(s))));
  };

  // The beach under each cell, once.
  const bedKernel = Fn(() => {
    const { i, j } = ij();
    bed.element(instanceIndex).assign(bedAt(cellX(i), cellY(j)));
  })().compute(COUNT);
  cleanup.push(() => bedKernel.dispose());

  // Still water over the beach, the sand wet below the waterline.
  const initKernel = Fn(() => {
    const z = bed.element(instanceIndex);
    const h = max(z.negate(), 0);
    for (const s of S) storage(s, "vec4", COUNT).element(instanceIndex).assign(vec4(h, 0, 0, 0));
    for (const e of E) storage(e, "vec4", COUNT).element(instanceIndex).assign(vec4(0, 0, smooth(0.08, -0.04, z), 0));
    for (const l of L) storage(l, "vec4", COUNT).element(instanceIndex).assign(vec4(0));
    If(instanceIndex.equal(0), () => { clock.element(0).assign(vec4(startTime, 0, 0, 0)); });
  })().compute(COUNT);
  cleanup.push(() => initKernel.dispose());

  /** Velocities: advect, add gravity, wet/dry walls, friction; the seaward face takes the incoming flow. */
  const momentum = (srcAttr: THREE.StorageBufferAttribute, dstAttr: THREE.StorageBufferAttribute) => {
    const src = storage(srcAttr, "vec4", COUNT).toReadOnly(), dst = storage(dstAttr, "vec4", COUNT);
    return Fn(() => {
      const { i, j } = ij();
      const t = clock.element(0).x;
      const iE = wrap(i.add(1)), iW = wrap(i.sub(1)), jS = clampJ(j.add(1)), jN = clampJ(j.sub(1));
      const s = load(src, i, j).toVar(), sE = load(src, iE, j).toVar(), sS = load(src, i, jS).toVar();
      const sW = load(src, iW, j).toVar(), sN = load(src, i, jN).toVar();
      const sNE = load(src, iE, jN), sSW = load(src, iW, jS);
      const z = bed.element(at(i, j)).sub(s.w).toVar(), zE = bed.element(at(iE, j)).sub(sE.w).toVar(), zS = bed.element(at(i, jS)).sub(sS.w).toVar();
      const north = select(j.greaterThan(0), float(1), float(0));
      const eta = z.add(s.x).toVar(), etaE = zE.add(sE.x), etaS = zS.add(sS.x);
      const faceX = vec2(i.add(1).mul(dx), cellY(j)).toVar(), faceY = vec2(cellX(i), j.add(1).mul(dx).add(SURF.Y0)).toVar();

      // u on the east face.
      const vbar = s.z.add(sE.z).add(sN.z.add(sNE.z).mul(north)).mul(0.25);
      const ua = bilerp(src, i.sub(s.y.mul(k)), j.sub(vbar.mul(k))).y.add(sE.y.add(sW.y).add(sS.y).add(sN.y).sub(s.y.mul(4)).mul(0.05)).toVar();
      const un = ua.sub(etaE.sub(eta).mul(G * dt / dx)).toVar();
      const wallX = s.x.lessThan(EPS).and(sE.x.lessThan(EPS)).or(s.x.lessThan(EPS).and(etaE.lessThan(z))).or(sE.x.lessThan(EPS).and(eta.lessThan(zE)));
      const hx = max(max(eta, etaE).sub(max(z, zE)), 0);
      un.assign(select(wallX, float(0), un));
      // v on the seaward face.
      const ubar = s.y.add(sW.y).add(sS.y).add(sSW.y).mul(0.25);
      const va = bilerp(src, i.sub(ubar.mul(k)), j.sub(s.z.mul(k))).z.add(sE.z.add(sW.z).add(sS.z).add(sN.z).sub(s.z.mul(4)).mul(0.05)).toVar();
      const vn = va.sub(etaS.sub(eta).mul(G * dt / dx)).toVar();
      const wallY = s.x.lessThan(EPS).and(sS.x.lessThan(EPS)).or(s.x.lessThan(EPS).and(etaS.lessThan(z))).or(sS.x.lessThan(EPS).and(eta.lessThan(zS)));
      const hy = max(max(eta, etaS).sub(max(z, zS)), 0);
      vn.assign(select(wallY, float(0), vn));
      // A finger dragged through the water pushes it along.
      for (const brush of brushes) {
        const wx = float(1).sub(smooth(0.3, 1, segment(faceX, brush).div(brushRadius))).mul(smooth(0.002, 0.02, hx));
        const wy = float(1).sub(smooth(0.3, 1, segment(faceY, brush).div(brushRadius))).mul(smooth(0.002, 0.02, hy));
        un.addAssign(push.x.sub(un).mul(wx).mul(0.15));
        vn.addAssign(push.y.sub(vn).mul(wy).mul(0.15));
      }
      // Manning friction (n = 0.025): thin sheets slow quickly, deeper water hardly at all.
      un.divAssign(un.abs().mul(G * dt * 0.025 * 0.025).div(max(hx, 0.003).pow(4 / 3)).add(1));
      vn.divAssign(vn.abs().mul(G * dt * 0.025 * 0.025).div(max(hy, 0.003).pow(4 / 3)).add(1));
      // The open sea beyond the last row.
      If(j.equal(NY - 1), () => { vn.assign(incoming(faceY.x, float(SURF.Y1), t.add(dt)).flow.negate()); });
      dst.element(instanceIndex).assign(vec4(s.x, clamp(un, -UMAX, UMAX), clamp(vn, -UMAX, UMAX), s.w));
    })().compute(COUNT);
  };

  /** Depth from the face fluxes; the incoming sea in the relaxation zone; channels; foam, sand, wetness, lace. */
  const continuity = (srcAttr: THREE.StorageBufferAttribute, dstAttr: THREE.StorageBufferAttribute,
    eFrom: THREE.StorageBufferAttribute, eTo: THREE.StorageBufferAttribute, lFrom: THREE.StorageBufferAttribute, lTo: THREE.StorageBufferAttribute) => {
    const src = storage(srcAttr, "vec4", COUNT).toReadOnly(), dst = storage(dstAttr, "vec4", COUNT);
    const e0 = storage(eFrom, "vec4", COUNT).toReadOnly(), e1 = storage(eTo, "vec4", COUNT);
    const l0 = storage(lFrom, "vec4", COUNT).toReadOnly(), l1 = storage(lTo, "vec4", COUNT);
    return Fn(() => {
      const { i, j } = ij();
      const t = clock.element(0).x.toVar();
      // Declared up front: a shared expression is first evaluated where it is first used, and here that is inside
      // the last row's branch, which would leave it unset for every other cell.
      const x = cellX(i).toVar(), y = cellY(j).toVar();
      const iE = wrap(i.add(1)), iW = wrap(i.sub(1)), jS = clampJ(j.add(1)), jN = clampJ(j.sub(1));
      const s = load(src, i, j).toVar(), sE = load(src, iE, j).toVar(), sS = load(src, i, jS).toVar();
      const sW = load(src, iW, j).toVar(), sN = load(src, i, jN).toVar();
      const z = bed.element(at(i, j)).sub(s.w).toVar();
      const zE = bed.element(at(iE, j)).sub(sE.w), zW = bed.element(at(iW, j)).sub(sW.w);
      const zS = bed.element(at(i, jS)).sub(sS.w), zN = bed.element(at(i, jN)).sub(sN.w);
      const last = j.equal(NY - 1), first = j.equal(0);
      const h = s.x, eta = z.add(h).toVar();
      const etaE = zE.add(sE.x), etaW = zW.add(sW.x), etaN = zN.add(sN.x);
      // Beyond the last row lies the incoming sea.
      const hS = sS.x.toVar(), etaS = zS.add(sS.x).toVar(), zSg = zS.toVar();
      If(last, () => {
        const ghost = incoming(x, float(SURF.Y1 + dx / 2), t).eta;
        hS.assign(max(ghost.sub(z), 0)); etaS.assign(ghost); zSg.assign(z);
      });
      const faceDepth = (hU: F, hD: F, eU: F, eD: F, zU: F, zD: F, vel: F) =>
        min(select(vel.greaterThan(0), hU, hD), max(select(vel.greaterThan(0), eU, eD).sub(max(zU, zD)), 0));
      const uE = s.y, uW = sW.y, vS = s.z, vN = select(first, float(0), sN.z);
      const fE = uE.mul(faceDepth(h, sE.x, eta, etaE, z, zE, uE));
      const fW = uW.mul(faceDepth(sW.x, h, etaW, eta, zW, z, uW));
      const gS = vS.mul(faceDepth(h, hS, eta, etaS, z, zSg, vS));
      const gN = vN.mul(faceDepth(sN.x, h, etaN, eta, zN, z, vN));
      const hn = max(h.sub(fE.sub(fW).add(gS).sub(gN).mul(k)), 0).toVar();
      const un = uE.toVar(), vn = vS.toVar();
      // Relaxation zone: the incoming sea, which also swallows what the beach sends back.
      const relax = clamp(y.sub(SURF.Y1 - SURF.SPONGE).div(SURF.SPONGE), 0, 1).pow(2).mul(0.25).toVar();
      If(relax.greaterThan(0), () => {
        const centre = incoming(x, y, t.add(dt));
        hn.addAssign(max(centre.eta.sub(z), 0).sub(hn).mul(relax));
        vn.addAssign(incoming(x, y.add(dx / 2), t.add(dt)).flow.negate().sub(vn).mul(relax));
        un.subAssign(un.mul(relax));
      });

      // Channels: a stroke on the sand digs up to 6 cm; moving water slumps the walls and fills them in.
      const cut = s.w.toVar();
      const p = vec2(x, y);
      for (const brush of brushes) {
        const carve = float(1).sub(smooth(0.35, 1, segment(p, brush).div(brushRadius))).mul(dig).mul(smooth(0.03, 0.005, h));
        cut.assign(max(cut, carve.mul(0.06)));
      }
      // Writing, read through the mip whose pixels are this cell's size, so the water runs into the letters.
      cut.mulAssign(smoothed(p).oneMinus());
      const lettering = float(1).sub(smooth(0.35, 1, inked(p, 2))).mul(dig).mul(smooth(0.03, 0.005, h));
      cut.assign(max(cut, lettering.mul(0.06)));
      const wetNow = smooth(0.002, 0.01, hn).toVar();
      const speed = length(vec2(uE.add(uW), vS.add(vN)).mul(0.5)).toVar();
      const around = sE.w.add(sW.w).add(sS.w).add(sN.w).mul(0.25);
      cut.addAssign(around.sub(cut).mul(dt).mul(wetNow.mul(0.6).add(0.02)));
      cut.mulAssign(exp(float(dt).mul(filling(wetNow, speed)).negate()));

      // Matter, carried by the flow.
      const uc = uE.add(uW).mul(0.5), vc = vS.add(vN).mul(0.5);
      const back = vec2(i.sub(uc.mul(k)), j.sub(vc.mul(k)));
      const carried = bilerp(e0, back.x, back.y).toVar();
      const here = e0.element(instanceIndex).toVar();
      const divergence = uE.sub(uW).add(vS).sub(vN).div(dx);
      // Foam is conserved on the surface: it piles up where the water converges and thins where it spreads.
      const foam = carried.x.mul(clamp(float(1).sub(divergence.mul(dt)), 0.5, 1.5)).toVar();
      // Breaking: a steep, rising front that stands high for the depth under it.
      const steep = length(vec2(etaE.sub(etaW), etaS.sub(etaN)).div(2 * dx));
      const rising = hn.sub(h).div(dt);
      // Inside the surf zone (still water under about 0.75 m for these waves) every steep, rising front is a broken
      // bore; a wave standing high for its depth breaks a little further out.
      const still = bed.element(at(i, j)).negate();
      const patch = noise(vec2(x.mul(1.4), y.add(t.mul(1.9)).mul(0.8))).mul(0.65).add(noise(vec2(x.mul(3.1).add(7.3), y.add(t.mul(1.9)).mul(2.2))).mul(0.35));
      const breaking = smooth(0.4, 0.6, patch).mul(0.92).add(0.08).mul(smooth(0.35, 0.8, steep)).mul(smooth(0.2, 1, rising)).mul(smooth(0.003, 0.02, hn))
        .mul(max(smooth(0.95, 0.6, still), smooth(0.25, 0.4, eta.div(max(hn, 0.02))))).mul(max(float(1).sub(relax.mul(4)), 0)).toVar();
      // The running edge of the swash (a wet cell beside dry sand) froths as it climbs.
      const edge = smooth(0.01, 0.002, min(min(sE.x, sW.x), min(sS.x, sN.x)));
      const climbing = smooth(0.3, 1.2, vc.negate()).mul(edge).mul(wetNow);
      // Fresh foam's big bubbles burst within a second, tearing the mat into lace; the lace lasts several seconds.
      foam.assign(min(foam.add(breaking.mul(16 * dt)).add(climbing.mul(3 * dt)), 1.5)
        .mul(exp(smooth(0.6, 1.4, foam).mul(1.6).add(0.2).mul(-dt))));
      // Stirred sand: thrown up by the bores and by fast, thin swash; it settles over several seconds.
      const sand = min(carried.y.mul(Math.exp(-dt / 6)).add(breaking.mul(1.6 * dt)).add(smooth(0.6, 1.8, speed).mul(smooth(0.15, 0.02, hn)).mul(wetNow).mul(0.5 * dt)), 1);
      // Wet sand dries over about twenty seconds once the water has gone.
      const wet = max(here.z.mul(Math.exp(-dt / 20)), wetNow);
      // Foam left on the sand when the water drains away; the next wave picks it up again.
      const wasWet = smooth(0.002, 0.01, h);
      const stranded = max(here.w.mul(Math.exp(-dt / 3.5)), foam.mul(min(max(wasWet.sub(wetNow), 0).mul(40), 1))).mul(wetNow.oneMinus()).toVar();
      foam.assign(max(foam.mul(wetNow), here.w.mul(wetNow)));

      // Lace displacement: where the pattern on the surface has been carried since its layer last reset.
      // The lace drifts a little slower than the water, so a fast backwash draws it into streaks, not threads.
      const drift = bilerp(l0, back.x, back.y).add(vec4(uc, vc, uc, vc).mul(0.7 * dt)).toVar();
      const phaseA = t.div(LACE_PERIOD), phaseB = t.div(LACE_PERIOD).add(0.5);
      const resetA = floor(phaseA).notEqual(floor(phaseA.sub(dt / LACE_PERIOD)));
      const resetB = floor(phaseB).notEqual(floor(phaseB.sub(dt / LACE_PERIOD)));
      drift.assign(vec4(select(resetA, vec2(0), drift.xy), select(resetB, vec2(0), drift.zw)));

      dst.element(instanceIndex).assign(vec4(hn, un, vn, cut));
      e1.element(instanceIndex).assign(vec4(foam, sand, wet, stranded));
      l1.element(instanceIndex).assign(drift);
    })().compute(COUNT);
  };

  const tick = Fn(() => { clock.element(0).assign(clock.element(0).add(vec4(dt, 0, 0, 0))); })().compute(1);
  cleanup.push(() => tick.dispose());

  /** The four textures the renderer reads. */
  const pack = (sAttr: THREE.StorageBufferAttribute, eAttr: THREE.StorageBufferAttribute, lAttr: THREE.StorageBufferAttribute) => {
    const s0 = storage(sAttr, "vec4", COUNT).toReadOnly(), e = storage(eAttr, "vec4", COUNT).toReadOnly(), l = storage(lAttr, "vec4", COUNT).toReadOnly();
    return Fn(() => {
      const { i, j } = ij();
      const iE = wrap(i.add(1)), iW = wrap(i.sub(1)), jS = clampJ(j.add(1)), jN = clampJ(j.sub(1));
      const s = load(s0, i, j), sE = load(s0, iE, j), sW = load(s0, iW, j), sS = load(s0, i, jS), sN = load(s0, i, jN);
      const eta = (c: V4, ii: F, jj: F) => bed.element(at(ii, jj)).sub(c.w).add(c.x);
      const e0 = eta(s, i, j);
      // The slope of a dry neighbour is the beach's own, so the water's edge has no false step.
      const pick = (c: V4, ii: F, jj: F) => select(c.x.greaterThan(EPS).or(s.x.lessThan(EPS)), eta(c, ii, jj), e0.add(bed.element(at(ii, jj)).sub(c.w).sub(bed.element(at(i, j)).sub(s.w))));
      const gx = pick(sE, iE, j).sub(pick(sW, iW, j)).div(2 * dx);
      const gy = pick(sS, i, jS).sub(pick(sN, i, jN)).div(select(j.equal(0).or(j.equal(NY - 1)), float(dx), float(2 * dx)));
      const p = ivec2(int(i), int(j));
      textureStore(surface, p, vec4(e0, gx, gy, s.x)).toWriteOnly();
      textureStore(drift, p, l.element(instanceIndex)).toWriteOnly();
      textureStore(matter, p, e.element(instanceIndex)).toWriteOnly();
    })().compute(COUNT);
  };

  const kernels = {
    momentum: momentum(S[0], S[1]),
    continuity: [continuity(S[1], S[0], E[0], E[1], L[0], L[1]), continuity(S[1], S[0], E[1], E[0], L[1], L[0])],
    pack: [pack(S[0], E[0], L[0]), pack(S[0], E[1], L[1])],
  };
  cleanup.push(() => { kernels.momentum.dispose(); kernels.continuity.forEach(n => n.dispose()); kernels.pack.forEach(n => n.dispose()); });

  // The drawn channels as they are seen. The water flows through the cells' coarse channels above; the groove a
  // finger leaves is drawn from this finer field (a rounded trench with the sand it pushed aside heaped along its
  // edges), which slumps and fills under the water at the solver's rates, scaled to its cells. It only runs once
  // something has been drawn.
  const GX = NX * 4, GY = NY * 4, GCOUNT = GX * GY, gdx = dx / 4;
  const grooves = [new THREE.StorageBufferAttribute(GCOUNT, 2), new THREE.StorageBufferAttribute(GCOUNT, 2)];
  const grooveTex = new THREE.StorageTexture(GX, GY);
  grooveTex.type = THREE.HalfFloatType; grooveTex.format = THREE.RGBAFormat;
  grooveTex.wrapS = THREE.RepeatWrapping; grooveTex.wrapT = THREE.ClampToEdgeWrapping;
  grooveTex.minFilter = grooveTex.magFilter = THREE.LinearFilter; grooveTex.generateMipmaps = false;
  cleanup.push(() => grooveTex.dispose());
  /** Seconds simulated since the groove last moved. */
  const grooveDt = uniform(0);
  const grooveKernel = (fromAttr: THREE.StorageBufferAttribute, toAttr: THREE.StorageBufferAttribute) => {
    const from = storage(fromAttr, "vec2", GCOUNT).toReadOnly(), to = storage(toAttr, "vec2", GCOUNT);
    const water = storage(S[0], "vec4", COUNT).toReadOnly();
    return Fn(() => {
      const gi = float(instanceIndex.mod(GX)).toVar(), gj = float(instanceIndex.div(GX)).toVar();
      const gwrap = (v: F) => v.sub(floor(v.div(GX)).mul(GX));
      const gat = (ii: F, jj: F) => int(clamp(jj, 0, GY - 1).mul(GX).add(gwrap(ii)));
      const c = from.element(instanceIndex).toVar();
      const cE = from.element(gat(gi.add(1), gj)).toVar(), cW = from.element(gat(gi.sub(1), gj)).toVar();
      const cS = from.element(gat(gi, gj.add(1))).toVar(), cN = from.element(gat(gi, gj.sub(1))).toVar();
      const p = vec2(gi.add(0.5).mul(gdx), gj.add(0.5).mul(gdx).add(SURF.Y0)).toVar();
      // The water over this point, from the solver's cells.
      const w = bilerp(water, p.x.div(dx).sub(0.5), p.y.sub(SURF.Y0).div(dx).sub(0.5)).toVar();
      const wet = smooth(0.002, 0.01, w.x).toVar();
      const speed = length(w.yz);
      const cut = c.x.toVar(), lip = c.y.toVar();
      // Slumping: the solver's rate per second, times sixteen for cells a quarter the size.
      const slump = min(grooveDt.mul(wet.mul(0.6).add(0.02)).mul(16), 0.9);
      cut.addAssign(cE.x.add(cW.x).add(cS.x).add(cN.x).mul(0.25).sub(cut).mul(slump));
      lip.addAssign(cE.y.add(cW.y).add(cS.y).add(cN.y).mul(0.25).sub(lip).mul(slump));
      cut.mulAssign(exp(grooveDt.mul(filling(wet, speed)).negate()));
      // Loose sand heaped on the surface goes faster than the trench below it.
      lip.mulAssign(exp(grooveDt.mul(wet.mul(0.6).add(speed.mul(speed).mul(wet).mul(1.6)).add(0.002)).negate()));
      const dry = smooth(0.03, 0.005, w.x).mul(dig).toVar();
      /** A finger's mark, d radii from its path: a rounded trench, as deep as the solver's, and a low heap along
       *  each side (none inside a trench already dug, or a stroke's joints would bead). */
      const mark = (d: F) => {
        const trench = float(1).sub(smooth(0.3, 1, d)).mul(dry);
        cut.assign(max(cut, trench.mul(0.06)));
        lip.assign(max(lip.mul(trench.oneMinus()), smooth(0.85, 1.15, d).mul(smooth(1.9, 1.15, d)).mul(dry).mul(0.01)
          .mul(smooth(0.012, 0, cut))));
      };
      for (const brush of brushes) mark(segment(p, brush).div(brushRadius).toVar());
      const flat = smoothed(p).oneMinus();
      cut.mulAssign(flat); lip.mulAssign(flat);
      mark(inked(p, 0).toVar());
      to.element(instanceIndex).assign(vec2(cut, lip));
      // Depth and slope for the renderer (the slope from the neighbours as they were a frame ago).
      const depth = (v: V2) => v.x.sub(v.y);
      const gx = depth(cE).sub(depth(cW)).div(2 * gdx), gy = depth(cS).sub(depth(cN)).div(2 * gdx);
      textureStore(grooveTex, ivec2(int(gi), int(gj)), vec4(cut.sub(lip), gx, gy, 0)).toWriteOnly();
    })().compute(GCOUNT);
  };
  const grooveKernels = [grooveKernel(grooves[0], grooves[1]), grooveKernel(grooves[1], grooves[0])];
  const grooveClear = Fn(() => {
    for (const g of grooves) storage(g, "vec2", GCOUNT).element(instanceIndex).assign(vec2(0));
    textureStore(grooveTex, ivec2(int(instanceIndex.mod(GX)), int(instanceIndex.div(GX))), vec4(0)).toWriteOnly();
  })().compute(GCOUNT);
  cleanup.push(() => { grooveKernels.forEach(n => n.dispose()); grooveClear.dispose(); });
  let grooveParity = 0, drawn = false;

  // The sound's measurements: for each row of cells and each band along the shore, the sum over the band's cells
  // of what makes each part of the sound of surf. The CPU adds up the rows.
  const ROWS = BANDS * NY, PER = NX / BANDS;
  const measured = new THREE.StorageBufferAttribute(ROWS, 4);
  const measureKernel = (eAttr: THREE.StorageBufferAttribute) => {
    const water = storage(S[0], "vec4", COUNT).toReadOnly(), e = storage(eAttr, "vec4", COUNT).toReadOnly();
    const out = storage(measured, "vec4", ROWS);
    return Fn(() => {
      const band = float(instanceIndex.mod(BANDS)).toVar(), j = float(instanceIndex.div(BANDS)).toVar();
      const sum = vec4(0).toVar();
      Loop(PER, ({ i }) => {
        const cell = at(band.mul(PER).add(float(i)), j);
        const s = water.element(cell).toVar(), m = e.element(cell).toVar();
        const wet = smooth(0.002, 0.01, s.x).toVar(), speed = length(s.yz).toVar();
        // A bore's roar: fresh foam where the water is running, the louder the faster.
        const roar = smooth(0.7, 1.3, m.x).mul(speed.add(0.3)).mul(wet);
        // The swash's hiss: a thin sheet of water running over the sand.
        const hiss = wet.mul(smooth(0.06, 0.01, s.x)).mul(smooth(0.1, 1, speed));
        // Fizz: old foam bursting, on the water and left on the sand.
        const fizz = smooth(0.08, 0.7, m.x).mul(smooth(1.2, 0.6, m.x)).mul(wet).add(m.w.mul(0.6));
        // The water's motion in the surf zone, for the low wash under it all.
        const swell = s.x.mul(speed).mul(smooth(SURF.Y1 - SURF.SPONGE, SURF.Y1 - SURF.SPONGE - 0.5, cellY(j)));
        sum.addAssign(vec4(roar, hiss, fizz, swell));
      });
      out.element(instanceIndex).assign(sum);
    })().compute(ROWS);
  };
  const measureKernels = [measureKernel(E[0]), measureKernel(E[1])];
  cleanup.push(() => measureKernels.forEach(n => n.dispose()));
  /** Full scale for each measure: band sums that make a loud surf. */
  const SCALE = [PER * NY * 0.03, PER * NY * 0.09, PER * NY * 0.12, PER * NY * 0.1];

  let parity = 0, steps = 0, started = 0;
  const run = (n: number) => {
    const list: THREE.ComputeNode[] = [];
    for (let s = 0; s < n; s++) {
      list.push(kernels.momentum, kernels.continuity[parity], tick);
      parity = 1 - parity; steps++;
    }
    list.push(kernels.pack[parity]);
    drawn ||= brushes.some(b => b.value.x > -90) || inkOn.value > 0;
    if (drawn) {
      grooveDt.value = n * dt;
      list.push(grooveKernels[grooveParity]);
      grooveParity = 1 - grooveParity;
    }
    renderer.compute(list);
  };
  const reset = (seconds = 0) => {
    startTime.value = seconds; started = seconds; steps = 0; parity = 0;
    grooveParity = 0; drawn = false;
    renderer.compute([bedKernel, initKernel, grooveClear]);
  };
  dig.value = 1;
  reset(0);

  return {
    surface, drift, matter, groove: grooveTex, nx: NX, ny: NY, dt,
    brushes, brushRadius, push,
    ink, inkScale: 1 / gdx, inkOn, inkTo, inkBox, inkWas,
    run,
    settle() {
      if (!drawn) return;
      grooveDt.value = 0;
      renderer.compute(grooveKernels[grooveParity]);
      grooveParity = 1 - grooveParity;
    },
    async measure() {
      renderer.compute(measureKernels[parity]);
      const rows = new Float32Array(await renderer.getArrayBufferAsync(measured));
      const bands = new Float32Array(BANDS * 4);
      for (let r = 0; r < ROWS; r++) for (let c = 0; c < 4; c++) bands[(r % BANDS) * 4 + c] += rows[r * 4 + c];
      for (let i = 0; i < bands.length; i++) bands[i] = Math.min(1.5, bands[i] / SCALE[i % 4]);
      return bands;
    },
    time: () => started + steps * dt,
    reset,
    dispose() {
      for (const release of cleanup.toReversed()) release();
    },
  };
}
