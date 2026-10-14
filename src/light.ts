import * as THREE from "three/webgpu";
import { clamp, exp, mix, pow, vec3, wgsl, wgslFn } from "three/tsl";

/**
 * The light over the beach as TSL building blocks, shared by the caustic pass and the material: the sun of a July
 * afternoon at 55° N, the absorption of clear coastal water, a sky dome and Fresnel. World units are screen heights;
 * y grows seaward (down the screen), x along the shore, z up to the camera, which looks straight down.
 */

type F = THREE.Node<"float">;
type V2 = THREE.Node<"vec2">;
type V3 = THREE.Node<"vec3">;

/** Metres of beach in one screen height. The sand's 2 m tile repeats 2.8 times in it. */
export const WORLD_M = 5.6;

/** The sun 55° up, a little west of south (2 pm in July at 55° N), as a unit vector towards it. */
export const SUN = (() => {
  const alt = 54.97 * Math.PI / 180, az = 202.07 * Math.PI / 180;
  return new THREE.Vector3(Math.sin(az) * Math.cos(alt), -Math.cos(az) * Math.cos(alt), Math.sin(alt)).normalize();
})();
export const sun = (): V3 => vec3(SUN.x, SUN.y, SUN.z);
/** The same sunlight after the flat surface has bent it, pointing down into the water. */
export const SUN_WATER = (() => {
  const i = SUN.clone().negate(), eta = 1 / 1.333, c = -i.z, k = 1 - eta * eta * (1 - c * c);
  return i.multiplyScalar(eta).add(new THREE.Vector3(0, 0, eta * c - Math.sqrt(k)));
})();

/** Shared WGSL: hash and value noise. */
const math = wgsl(/* wgsl */ `
fn swashHash(p0: vec2f) -> f32 {
  var p = fract(p0 * vec2f(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
fn swashNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(swashHash(i), swashHash(i + vec2f(1.0, 0.0)), u.x),
    mix(swashHash(i + vec2f(0.0, 1.0)), swashHash(i + vec2f(1.0)), u.x), u.y);
}
`);

const noiseFn = wgslFn(/* wgsl */ `fn swashNoiseAt(p: vec2f) -> f32 { return swashNoise(p); }`, [math]);
const hashFn = wgslFn(/* wgsl */ `fn swashHashAt(p: vec2f) -> f32 { return swashHash(p); }`, [math]);
export const noise = (p: V2) => noiseFn(p) as unknown as F;
export const hash = (p: V2) => hashFn(p) as unknown as F;

/** Water, per metre (R, G, B): attenuation of clear coastal water, and the colour its in-scatter tends to. Fitted
 * so 1 m over sand reads pale aqua, 2 m turquoise and the 4.4 m deep water a dense blue-teal. */
export const ABSORB = [0.55, 0.09, 0.06] as const;
export const DEEP = [0.005, 0.04, 0.12] as const;
export const absorb = (metres: F) => exp(vec3(...ABSORB).mul(metres).negate());

/** Clear July sky, linear, scaled against sunlit sand (= 1): a deep zenith and a pale horizon. */
export function sky(dir: V3): V3 {
  const up = clamp(dir.z, 0, 1);
  return mix(vec3(0.38, 0.47, 0.56), vec3(0.06, 0.14, 0.34), pow(up, 0.6));
}

/** Schlick's Fresnel for water seen along the surface normal's tilt. */
export const fresnel = (cosine: F) => pow(cosine.oneMinus(), 5).mul(0.98).add(0.02);
