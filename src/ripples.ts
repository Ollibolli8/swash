import * as THREE from "three/webgpu";
import { Fn, cos, float, instanceIndex, int, ivec2, texture, textureStore, uniform, vec2, vec4 } from "three/tsl";
import { WORLD_M } from "./light";

type V2 = THREE.Node<"vec2">;
type V4 = THREE.Node<"vec4">;

/**
 * The wind's ripples, the centimetre waves the surf solver's cells are too coarse for: a light onshore breeze's
 * spectrum (wavelengths 2.2 to 55 cm, spread either side of the wind, the longer ones steeper), summed on the GPU
 * every frame into a tileable slope texture. Each component runs at its true speed, capillary waves included
 * (omega² = g k + sigma/rho k³), so the caustics on the sand dance as fast as they do on a real beach. Ripples
 * shorter than SHORT are kept apart: they glint on the surface, but their light comes to a focus within a few
 * centimetres of water and blurs out (the sun is half a degree wide), so only the longer ones draw the caustic net
 * on a deeper bed. The texture is read twice, at two scales and angles, so its tile never shows.
 */

/** Metres of beach per tile. */
const TILE = 1.1;
const SIZE = 256;
const COUNT = 48;
/** Root-mean-square slope of the whole field: a light breeze, Cox and Munk's figure for about 2 m/s. */
const SLOPE_RMS = 0.075;
/** Metres: the boundary between glinting ripples and caustic-drawing waves. */
const SHORT = 0.07;
/** The second read: scale and angle against the first. */
const SCALE = 0.77, TURN = 0.61;
const TAU = Math.PI * 2;

const MODES = (() => {
  let seed = 0x51ed;
  const random = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const taken = new Set<string>(), modes: Array<{ kx: number; ky: number; a: number; w: number; p: number; short: boolean }> = [];
  while (modes.length < COUNT) {
    const wavelength = 0.022 * Math.pow(0.55 / 0.022, random());
    // Mostly downwind (towards the beach, -y), a few across it.
    const spread = (random() + random() + random() - 1.5) * 1.5;
    const n = Math.round(TILE / wavelength * Math.sin(spread)), m = Math.round(-TILE / wavelength * Math.cos(spread));
    if (taken.has(`${n},${m}`) || n * n + m * m < 4) continue;
    taken.add(`${n},${m}`);
    const kx = TAU * n / TILE, ky = TAU * m / TILE, k = Math.hypot(kx, ky);
    modes.push({ kx, ky, a: Math.pow(TAU / k / 0.1, 0.5) / k, w: Math.sqrt(9.81 * k + 7.28e-5 * k * k * k), p: random() * TAU, short: TAU / k < SHORT });
  }
  // Scale to the field's slope.
  const rms = Math.sqrt(modes.reduce((sum, m) => sum + (m.a * Math.hypot(m.kx, m.ky)) ** 2 / 2, 0));
  for (const m of modes) m.a *= SLOPE_RMS / rms;
  return modes;
})();

export function createRipples(renderer: THREE.WebGPURenderer): { texture: THREE.StorageTexture; update: (t: number) => void; dispose: () => void } {
  const map = new THREE.StorageTexture(SIZE, SIZE);
  map.type = THREE.HalfFloatType; map.format = THREE.RGBAFormat;
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.minFilter = map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = false;
  const time = uniform(0);
  const kernel = Fn(() => {
    const x = int(instanceIndex.mod(SIZE)), y = int(instanceIndex.div(SIZE));
    const p = vec2(x, y).add(0.5).mul(TILE / SIZE).toVar();
    // (slope x, slope y) of the long waves, then of the short ripples.
    const long = vec2(0).toVar(), short = vec2(0).toVar();
    for (const mode of MODES) {
      const slope = cos(p.x.mul(mode.kx).add(p.y.mul(mode.ky)).sub(time.mul(mode.w)).add(mode.p)).mul(mode.a);
      (mode.short ? short : long).addAssign(vec2(mode.kx, mode.ky).mul(slope));
    }
    textureStore(map, ivec2(x, y), vec4(long, short)).toWriteOnly();
  })().compute(SIZE * SIZE);
  return {
    texture: map,
    update(t) { time.value = t; renderer.compute(kernel); },
    dispose() { kernel.dispose(); map.dispose(); },
  };
}

/** The ripples' slope at world point w (screen heights): (long waves xy, short ripples xy). */
export function ripplesAt(map: THREE.Texture, w: V2): V4 {
  const p = w.mul(WORLD_M / TILE).toVar();
  const a = texture(map, p).level(float(0)) as unknown as V4;
  const c = Math.cos(TURN), s = Math.sin(TURN);
  const q = vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c))).mul(SCALE).add(0.37);
  const b = texture(map, q).level(float(0)) as unknown as V4;
  // The second read's slopes, turned back into the first's frame (and gentler: its waves are longer).
  const turn = (g: V2) => vec2(g.x.mul(c).add(g.y.mul(s)), g.y.mul(c).sub(g.x.mul(s))).mul(SCALE);
  return vec4(a.xy.mul(0.8).add(turn(b.xy).mul(0.6)), a.zw.mul(0.8).add(turn(b.zw).mul(0.6)));
}
