import * as THREE from "three/webgpu";
import { Fn, If, abs, clamp, cos, exp, float, floor, fract, log2, max, min, mix, normalize, pow, reflect, refract, sRGBTransferEOTF, screenSize, sin, smoothstep, texture, uv, vec2, vec3, vec4 } from "three/tsl";
import { foamCover, LACE_TILE } from "./foam";
import { DISPERSION, offStrip, stripUV, toMetres } from "./optics";
import { ripplesAt } from "./ripples";
import { ABSORB, DEEP, SUN, SUN_WATER, WORLD_M, fresnel, hash, noise, sky, sun } from "./light";
import { LACE_PERIOD, SURF, bedAt, incoming } from "./surf";

type F = THREE.Node<"float">;
type V2 = THREE.Node<"vec2">;
type V3 = THREE.Node<"vec3">;
type V4 = THREE.Node<"vec4">;

/** Display sRGB to linear light. */
const linear = (c: V3) => sRGBTransferEOTF(c) as unknown as V3;

/** An explicit-level sample: legal inside branches, where WGSL forbids implicit derivatives. */
const sample = (map: THREE.Texture, at: V2, level: F | number = 0): V4 =>
  texture(map, at).level(typeof level === "number" ? float(level) : level) as unknown as V4;

/** A cubic B-spline sample from four bilinear taps: the solver's cells blend into smooth curves (a water's edge
 * that would step from cell to cell rounds off, and the surface's slope has no creases at cell borders). */
function smoothSample(map: THREE.Texture, at: V2, nx: number, ny: number): V4 {
  const size = vec2(nx, ny);
  const st = at.mul(size).sub(0.5).toVar();
  const i = floor(st), f = st.sub(i).toVar();
  const f2 = f.mul(f), f3 = f2.mul(f);
  const w0 = vec2(1).sub(f).pow(3).div(6), w1 = f3.mul(3).sub(f2.mul(6)).add(4).div(6);
  const w2 = f2.mul(3).add(f.mul(3)).sub(f3.mul(3)).add(1).div(6), w3 = f3.div(6);
  const g0 = w0.add(w1), g1 = w2.add(w3);
  const h0 = w1.div(g0).sub(1).add(i).add(0.5).div(size), h1 = w3.div(g1).add(1).add(i).add(0.5).div(size);
  const a = sample(map, vec2(h0.x, h0.y)), b = sample(map, vec2(h1.x, h0.y)), c = sample(map, vec2(h0.x, h1.y)), d = sample(map, vec2(h1.x, h1.y));
  return a.mul(g0.x).add(b.mul(g1.x)).mul(g0.y).add(c.mul(g0.x).add(d.mul(g1.x)).mul(g1.y));
}

export type SwashUniforms = {
  time: THREE.UniformNode<"float", number>;
  aspect: THREE.UniformNode<"float", number>;
  /** World y at the top of the view (screen heights). */
  cam: THREE.UniformNode<"float", number>;
  sand: THREE.Texture;
  normal: THREE.Texture;
  roughness: THREE.Texture;
  caustics: THREE.Texture;
  lace: THREE.Texture;
  /** The wind's ripples (ripples.ts). */
  ripples: THREE.Texture;
  /** Rings from the pointer (wake.ts): (height, slope x, slope y) on the strip. */
  wake: THREE.Texture;
  /** The solver's textures (surf.ts). */
  surface: THREE.Texture;
  drift: THREE.Texture;
  matter: THREE.Texture;
  /** The drawn channels: (depth below the beach, its slope x, slope y), on a grid four times finer. */
  groove: THREE.Texture;
  /** The solver's grid, cells along and across the shore. */
  cells: readonly [number, number];
  debugView?: boolean;
};

/** Mean of the Poly Haven sand albedo (display sRGB): the texture only supplies variation around the authored sand. */
const MEAN: [number, number, number] = [0.423358244, 0.376947339, 0.308730724];
const TAU = Math.PI * 2;
/** Wave ripple marks on the bed, world units (about 13 cm). */
const RIPPLE = 0.024;
/** The foam lace's tile, metres. */
const LACE_M = LACE_TILE * WORLD_M;
/** Sand stirred into the water: extinction per metre at full load, and the colour it scatters (linear). */
const SILT_EXT = [1.3, 1.2, 1.25] as const;
const SILT = [0.42, 0.42, 0.33] as const;

/** A highlight shoulder: linear light above 0.85 rolls off towards 1 instead of clipping. */
const shoulder = (c: V3) => {
  const over = max(c.sub(0.85), vec3(0));
  return min(c, vec3(0.85)).add(vec3(1).sub(exp(over.div(-0.15))).mul(0.15));
};

/**
 * Straight down on a July afternoon, over the simulated surf. Sand keeps its authored display-sRGB look; everything
 * wet is lit in linear light. Sunlight, bent and focused by the simulated surface (the caustic pass), and the sky
 * reach the sand bed through Baltic water clouded by the sand the bores stir up, come back up through it, and meet
 * the sky's reflection and the sun's highlight at the surface. Each wave's height darkens the water over the bed as
 * it passes; its front refracts the bed and throws a line of light. Foam rides on top in a lace carried by the
 * flow, lit by the sun across the waves' own slopes, with bubbles under it and its shadow on the sand below.
 */
export function createMaterial(u: SwashUniforms): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, toneMapped: false });
  material.fragmentNode = Fn(() => {
    // QuadMesh's WebGPU screen UV is top-down, like the world: the beach is at the top.
    const screen = uv().toVar();
    const t = u.time as unknown as F;
    const aspect = u.aspect as unknown as F;
    const w = vec2(screen.x.mul(aspect), u.cam.add(screen.y)).toVar();
    const pixel = float(1).div(max(screenSize.y, 1)).toVar();
    const m = toMetres(w, aspect).toVar();
    const q = stripUV(m).toVar();

    // The solver: surface (height, slope x, slope y, depth), matter (foam, stirred sand, wet sand, stranded foam),
    // drift (where the lace has been carried), and the channels drawn in the sand.
    const [nx, ny] = u.cells;
    // Foam, wetness and stirred sand are read through a small fixed warp (±3.5 cm): their edges take the ragged line
    // of the real thing rather than the solver's smooth curves.
    const warp = vec2(noise(m.mul(6.5)), noise(m.mul(6.5).add(3.3))).sub(0.5).mul(0.07).div(vec2(SURF.W, SURF.Y1 - SURF.Y0));
    const S = smoothSample(u.surface, q, nx, ny).toVar(), M = smoothSample(u.matter, q.add(warp), nx, ny).toVar();
    // The channels' edges crumble a few millimetres either way, as dug sand does.
    const crumble = vec2(noise(m.mul(38)), noise(m.mul(38).add(5.1))).sub(0.5).mul(0.008).div(vec2(SURF.W, SURF.Y1 - SURF.Y0));
    const D = sample(u.drift, q).toVar(), Gr = sample(u.groove, q.add(crumble)).toVar();
    const off = offStrip(m.y).toVar();
    const onStrip = off.oneMinus().toVar();
    const eta = S.x.toVar(), slope = S.yz.toVar();
    const gentle = (g: V2) => g.div(g.length().mul(2.2).add(1));
    // Offshore, the incoming sea the solver is fed with.
    If(off.greaterThan(0), () => {
      const sea = incoming(m.x, m.y, t);
      eta.assign(mix(eta, sea.eta, off)); slope.assign(mix(slope, sea.slope, off));
    });
    const channel = Gr.x.mul(onStrip).toVar();
    const bed = bedAt(m.x, m.y).sub(channel).toVar();
    const depth = eta.sub(bed).toVar();
    const foamAmount = M.x.mul(onStrip).toVar(), silt = M.y.mul(onStrip).toVar();
    const wetSand = M.z.mul(onStrip).toVar(), stranded = M.w.mul(onStrip).toVar();

    // Water's edge, a fraction of a pixel wide: the depth from the interpolated surface over the exact beach.
    const edgeSoft = max(pixel.mul(WORLD_M).mul(0.06), 0.0008);
    const sea = smoothstep(0.0012, edgeSoft.add(0.0012), depth).toVar();
    const water = sea;
    const film = max(depth, 0).toVar();
    const wetness = max(wetSand, water).toVar();

    // Sand, as authored: fine ripples filtered by the actual render pixel (no moiré on phones or Retina).
    const sandUV = w.mul(2.8).toVar();
    const sandLOD = max(0, log2(max(pixel.mul(2.8 * 1024), 1)).sub(0.65)).toVar();
    const sandDetail = sample(u.sand, sandUV, sandLOD).rgb.div(vec3(...MEAN)).toVar();
    const scan = sample(u.normal, sandUV, sandLOD).xyz.mul(2).sub(1).toVar();
    const rough = sample(u.roughness, sandUV, sandLOD).r.toVar();
    const ripplePatch = smoothstep(0.34, 0.69, noise(w.mul(vec2(5, 8)))).toVar();
    const sandPhase = w.x.mul(0.35).add(w.y).mul(noise(w.mul(3)).mul(45).add(185))
      .add(noise(w.mul(7)).mul(4.6)).add(noise(w.mul(14.2).add(7.1)).mul(2.3)).toVar();
    // The channel's walls: the slope of its depth, stored with it, so the light across them is smooth.
    const cutSlope = Gr.yz.mul(onStrip).toVar();
    const groove = smoothstep(0, 0.05, channel).toVar();
    const sandSlope = cos(sandPhase).mul(0.028).mul(ripplePatch).mul(wetness.mul(0.72).oneMinus());
    const sandNormal = normalize(vec3(vec2(0.35, 1).mul(sandSlope).sub(cutSlope.mul(0.6))
      .add(vec2(scan.x, scan.y.negate()).mul(mix(0.24, 0.12, wetness))).negate(), 1)).toVar();
    const light = clamp(sandNormal.dot(sun()).div(SUN.z), 0.7, 1.2).toVar();
    const drySand = vec3(0.942, 0.891, 0.795).mul(sandDetail.mul(0.54).add(0.46))
      .mul(sin(sandPhase).mul(ripplePatch).mul(0.014).add(1)).toVar();
    // Sand stays dark while its pores hold water, then pales quickly as they drain: a soft but definite line.
    const moisture = smoothstep(0.15, 0.6, wetness.add(noise(w.mul(vec2(24, 68))).sub(0.5).mul(0.08))).toVar();
    // A channel's floor is the damper, darker sand from under the surface.
    const sandColour = mix(drySand, drySand.mul(vec3(0.8, 0.765, 0.69)), moisture).mul(light.mul(0.3).add(0.7))
      .mul(mix(vec3(1), vec3(0.86, 0.82, 0.76), groove)).toVar();
    // Damp sand just uncovered holds a film of water that mirrors the sky and the sun.
    const half = normalize(sun().add(vec3(0, 0, 1)));
    const wetSheen = pow(max(sandNormal.dot(half), 0), mix(250, 65, rough));
    const glaze = smoothstep(0.93, 1, wetSand).mul(water.oneMinus());
    sandColour.addAssign(vec3(0.15, 0.16, 0.14).mul(wetSheen).mul(moisture.add(glaze.mul(0.6))).mul(water.oneMinus()));
    sandColour.assign(mix(sandColour, sandColour.mul(vec3(0.9, 0.88, 0.85)).add(vec3(0.012, 0.016, 0.022)), glaze));
    const colour = linear(clamp(sandColour, vec3(0), vec3(1))).toVar();

    // Water, in linear light.
    If(water.greaterThan(0.001), () => {
      // The surface: the simulated waves, with the wind's centimetre ripples on them (fewer in foam and thin sheets).
      const ripples = ripplesAt(u.ripples, w);
      const wind = ripples.xy.add(ripples.zw);
      const ruffle = smoothstep(0.003, 0.03, film).mul(smoothstep(1.2, 0.2, foamAmount).mul(0.6).add(0.4));
      const waveSlope = gentle(slope).mul(sea);
      // The pointer's rings, smoothed out by thick foam.
      const hand = sample(u.wake, q).yz.mul(onStrip).mul(smoothstep(1.2, 0.3, foamAmount));
      const normal = normalize(vec3(waveSlope.add(wind.mul(ruffle)).add(hand.mul(sea)).negate(), 1)).toVar();
      const view = refract(vec3(0, 0, -1), normal, 1 / 1.333).toVar();
      const bedW = w.add(view.xy.mul(film.div(WORLD_M)).div(max(view.z.negate(), 0.2))).toVar();
      const bedM = toMetres(bedW, aspect);

      // The bed: the same sand, wet, with wave ripple marks offshore, blurred by the water above it.
      const bedUV = bedW.mul(2.8);
      const bedDetail = sample(u.sand, bedUV, sandLOD.add(film.mul(0.45))).rgb.div(vec3(...MEAN));
      const bedScan = sample(u.normal, bedUV, sandLOD.add(0.3).add(film.mul(0.3))).xyz.mul(2).sub(1);
      const rippleK = TAU / RIPPLE;
      const ripplePhase = bedW.y.add(noise(bedW.mul(vec2(3.1, 9.3))).mul(0.032)).add(sin(bedW.x.mul(29)).mul(0.0035)).mul(rippleK).toVar();
      const rippleAmp = smoothstep(0.25, 0.9, film).mul(noise(bedW.mul(vec2(4.3, 2.1))).mul(0.65).add(0.35))
        .mul(smoothstep(0.004, 0.0016, pixel)).mul(exp(film.div(-3.2)));
      // Sharp crests, wide troughs: sin + sin(2x)/4, differentiated.
      const rippleRise = cos(ripplePhase).add(cos(ripplePhase.mul(2)).mul(0.5)).mul(rippleAmp).mul(0.11);
      const bedNormal = normalize(vec3(bedScan.x.mul(-0.22).add(cutSlope.x.mul(0.6)), bedScan.y.mul(0.22).sub(rippleRise).add(cutSlope.y.mul(0.6)), 1));
      const damp = linear(vec3(0.942 * 0.76, 0.891 * 0.72, 0.795 * 0.64).mul(bedDetail.mul(0.6).add(0.4))
        .mul(sin(ripplePhase).mul(rippleAmp).mul(0.03).add(1))).toVar();
      const bedAlbedo = mix(damp, vec3(damp.dot(vec3(0.2126, 0.7152, 0.0722)).mul(1.2)), smoothstep(0.3, 2.5, film)).toVar();

      // Sunlight reaching the bed: focused by the surface, dimmed by the water and the sand in it, shaded by the
      // ripples and by the foam above (its shadow falls along the sun's path through the water).
      const toBed = vec2(bedW.x.div(aspect), bedW.y.sub(u.cam)).toVar();
      const spread = vec2(-SUN.x, -SUN.y).normalize().mul(film.div(WORLD_M).mul(DISPERSION / 2)).div(vec2(aspect, 1)).toVar();
      const caustic = vec3(sample(u.caustics, toBed.sub(spread)).r, sample(u.caustics, toBed).r, sample(u.caustics, toBed.add(spread)).r);
      const clouded = exp(silt.mul(film).mul(-3)).toVar();
      // The sun is half a degree wide: the net's cells never go dark, and its lines soften with depth.
      const focus = mix(vec3(1), clamp(caustic, vec3(0.4), vec3(mix(2.4, 4, smoothstep(0.02, 0.2, film)))), smoothstep(0, 0.03, film).mul(clouded)
        .mul(smoothstep(1.1, 0.3, foamAmount))
        .div(film.div(1.6).add(1)));
      const cosT = -SUN_WATER.z;
      const toSun = bedM.sub(vec2(SUN_WATER.x, SUN_WATER.y).mul(film.div(cosT)));
      const shade = sample(u.matter, stripUV(toSun)).x.mul(offStrip(toSun.y).oneMinus());
      const foamShadow = float(1).sub(smoothstep(0.2, 1.2, shade).mul(0.45).mul(smoothstep(0.03, 0.35, film)));
      const relief = max(bedNormal.dot(vec3(-SUN_WATER.x, -SUN_WATER.y, cosT)), 0).div(cosT);
      const extinction = vec3(...ABSORB).add(vec3(...SILT_EXT).mul(silt)).toVar();
      const up = exp(extinction.mul(film).negate()).toVar();
      const down = exp(extinction.mul(film.div(cosT)).negate());
      // Warm sun, blue sky: the shares sunlit sand is authored under.
      const reach = down.mul(focus).mul(relief).mul(foamShadow).mul(vec3(0.8, 0.78, 0.745))
        .add(exp(extinction.mul(film.mul(1.2)).negate()).mul(vec3(0.12, 0.16, 0.23)));
      // In-scatter: the clear water's blue-teal, turning milky green-sand where the bores have stirred the bed.
      const scatter = mix(vec3(...DEEP), vec3(...SILT), silt.div(silt.add(0.3)));
      const under = bedAlbedo.mul(reach).mul(up).add(scatter.mul(up.mul(down).oneMinus())).toVar();
      // Bubbles carried down under fresh foam glow pale through the water.
      const aerated = smoothstep(0.2, 1.3, foamAmount).mul(smoothstep(0.02, 0.25, film)).mul(0.55);
      under.assign(mix(under, vec3(0.5, 0.68, 0.66), aerated));

      // The surface: the sky in a tilted mirror, mostly transparent from straight above; the sun's highlight.
      const mirror = fresnel(normal.z).toVar();
      const reflected = sky(reflect(vec3(0, 0, -1), normal));
      // GGX with the roughness of the ripples too small to draw (a light breeze, Cox and Munk), so the highlight is
      // a soft sheen on the faces turned to the sun, not pixel glitter.
      const nh = max(normal.dot(half), 0);
      const a2 = float(0.012);
      const ggx = a2.div(nh.mul(nh).mul(a2.sub(1)).add(1).pow(2).mul(Math.PI));
      const highlight = ggx.mul(fresnel(max(sun().dot(half), 0))).mul(1.2).mul(smoothstep(0.002, 0.02, film));
      const surface = under.mul(mirror.oneMinus()).add(reflected.mul(mirror)).add(vec3(1, 0.96, 0.88).mul(highlight)).toVar();
      colour.assign(mix(colour, surface, water));
    });

    // Foam: the lace the flow has carried, the bore's dense roller, and foam left on the sand.
    If(max(foamAmount, stranded).greaterThan(0.01), () => {
      // Two copies of the lace, each carried by the flow for LACE_PERIOD and then handed over to the other.
      const phase = t.div(LACE_PERIOD).toVar();
      const weightA = smoothstep(0.15, 0.85, float(1).sub(abs(fract(phase).mul(2).sub(1)))).toVar();
      const offsetA = vec2(hash(vec2(floor(phase), 1.7)), hash(vec2(floor(phase), 8.3))).mul(9);
      const offsetB = vec2(hash(vec2(floor(phase.add(0.5)), 4.1)), hash(vec2(floor(phase.add(0.5)), 2.9))).mul(9);
      const laceA = sample(u.lace, m.sub(D.xy).div(LACE_M).add(offsetA)).toVar();
      const laceB = sample(u.lace, m.sub(D.zw).div(LACE_M).add(offsetB)).toVar();
      const lace = mix(laceB, laceA, weightA).toVar();
      const soft = pixel.mul(WORLD_M / LACE_M * 1.4).toVar();
      // How far from its strands the foam reaches: a fresh mat closes nearly every hole; old foam is thin strands.
      const amount = clamp(foamAmount, 0, 1).toVar();
      const fresh = smoothstep(0.85, 1.35, foamAmount).toVar();
      const reach = pow(amount, 1.25).mul(0.82).mul(smoothstep(0.2, 0.8, lace.z).mul(1.1).add(0.45)).add(fresh.mul(0.15)).toVar();
      const feather = soft.add(amount.oneMinus().mul(0.07)).add(0.04).toVar();
      const strands = smoothstep(reach.add(feather), reach.sub(feather), lace.x).mul(smoothstep(0, 0.06, amount)).toVar();
      // Foam is never quite opaque: thin strands let the water through, and a milky veil hangs between them.
      const opacity = strands.mul(mix(0.5, 0.97, smoothstep(0.15, 0.9, amount).max(fresh))).toVar();
      const veil = smoothstep(0.05, 0.9, amount).mul(lace.z.mul(0.8).add(0.6)).mul(0.2);
      opacity.assign(max(opacity, veil).add(lace.y.mul(strands).mul(0.15)));
      // Foam the water left on the sand, fixed to the beach.
      const sandLace = sample(u.lace, m.div(vec2(LACE_M, LACE_M * 0.55)).add(vec2(5.3, 2.9)));
      const onSand = foamCover(sandLace, stranded.mul(0.85), soft.add(0.05)).mul(0.8).toVar();
      // Lit across the waves' own slopes (gently: a bore's face is mostly hidden under its roller), and across its
      // own lumps: the lace's strands are ridges, with a side to the sun and a side away.
      const ridge = vec2(sample(u.lace, m.sub(D.xy).div(LACE_M).add(offsetA).add(vec2(0.01, 0))).x.sub(laceA.x),
        sample(u.lace, m.sub(D.xy).div(LACE_M).add(offsetA).add(vec2(0, 0.01))).x.sub(laceA.x)).div(0.01).mul(weightA)
        .add(vec2(sample(u.lace, m.sub(D.zw).div(LACE_M).add(offsetB).add(vec2(0.01, 0))).x.sub(laceB.x),
          sample(u.lace, m.sub(D.zw).div(LACE_M).add(offsetB).add(vec2(0, 0.01))).x.sub(laceB.x)).div(0.01).mul(weightA.oneMinus()));
      const lumps = ridge.mul(0.006).mul(smoothstep(0, 0.6, amount)).mul(float(1).sub(fresh.mul(0.5)));
      const foamNormal = normalize(vec3(gentle(slope).mul(-0.8).add(lumps), 1));
      const lit = max(foamNormal.dot(sun()), 0).mul(0.8).add(0.2).div(SUN.z * 0.8 + 0.2);
      // Brightest in its thick middle, greyer and bluer where it is thin and the water shows through.
      const depthIn = smoothstep(reach.sub(0.02), reach.sub(0.3), lace.x).mul(amount.add(fresh)).min(1);
      const foam = mix(linear(vec3(0.8, 0.86, 0.87)), linear(vec3(0.95, 0.96, 0.955)), depthIn.mul(0.7).add(0.3))
        .mul(lace.z.mul(0.08).add(0.95)).mul(lit).toVar();
      // Drying foam on sand is a little dingier.
      foam.assign(mix(foam, foam.mul(vec3(0.93, 0.91, 0.86)), onSand.mul(water.oneMinus())));
      const cover = max(opacity.mul(water), onSand).toVar();
      colour.assign(mix(colour, foam, clamp(cover, 0, 1)));
    });

    // `?surf-view`: the solver's fields in false colour (red foam, green surface height ±0.25 m, blue slope to 1).
    if (u.debugView) {
      return vec4(min(foamAmount, 1), clamp(eta.mul(2).add(0.5), 0, 1).mul(sea), clamp(slope.length(), 0, 1).mul(sea), 1);
    }
    return vec4(shoulder(colour), 1);
  })();
  return material;
}
