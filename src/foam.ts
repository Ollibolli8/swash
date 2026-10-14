import * as THREE from "three/webgpu";
import { Fn, instanceIndex, int, ivec2, max, pow, smoothstep, textureStore, vec2, wgsl, wgslFn } from "three/tsl";

type F = THREE.Node<"float">;
type V4 = THREE.Node<"vec4">;

/** World units per tile of the lace (about 2.4 m of beach). */
export const LACE_TILE = 0.42;
const SIZE = 512;

/**
 * A tileable foam lace, generated once on the GPU. The idea is Dan Greenheck's in Tidewater (MIT): store the
 * distance to the nearest bubble strand rather than a picture of foam, then threshold it by how much foam there is.
 * Plenty of foam is a dense mat; as it ages it tears into holes, then lace, then single strands, then nothing.
 *   R: distance to a strand (0 on it, ~1 in the middle of a hole)  G: small bubbles along the strands  B: mottling
 */
const laceMath = wgsl(/* wgsl */ `
fn laceHash(p: vec2f) -> vec2f {
  return fract(sin(vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)))) * 43758.5453);
}
fn laceWrap(x: vec2f, n: f32) -> vec2f { return x - n * floor(x / n); }
fn laceNoise(uv: vec2f, n: f32) -> f32 {
  let p = uv * n;
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = laceHash(laceWrap(i, n)).x;
  let b = laceHash(laceWrap(i + vec2f(1.0, 0.0), n)).x;
  let c = laceHash(laceWrap(i + vec2f(0.0, 1.0), n)).x;
  let d = laceHash(laceWrap(i + vec2f(1.0, 1.0), n)).x;
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn laceFbm(uv: vec2f, n: f32) -> f32 {
  return (laceNoise(uv, n) * 0.5 + laceNoise(uv, n * 2.0) * 0.25 + laceNoise(uv, n * 4.0) * 0.125) / 0.875;
}
// Periodic Worley: nearest and second-nearest feature distances, in cells.
fn laceCells(uv: vec2f, n: f32) -> vec2f {
  let p = uv * n;
  let ip = floor(p);
  let fp = fract(p);
  var f1 = 8.0;
  var f2 = 8.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let o = vec2f(f32(i), f32(j));
      let h = laceHash(laceWrap(ip + o, n)) * 0.85 + 0.075;
      let d = length(o + h - fp);
      if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
    }
  }
  return vec2f(f1, f2);
}
fn laceAt(uv: vec2f) -> vec4f {
  // Two-level warp: organic strands, never a honeycomb.
  let warp = (vec2f(laceFbm(uv, 3.0), laceFbm(uv + 0.37, 3.0)) - 0.5) * 0.3;
  let wuv = uv + warp + (vec2f(laceFbm(uv + 0.11, 11.0), laceFbm(uv + 0.83, 11.0)) - 0.5) * 0.07;
  // Half of (F2 - F1) is the distance to the cell edge: the strands are the edges of two nested networks.
  let big = laceCells(wuv, 7.0);
  let small = laceCells(wuv + 0.21, 17.0);
  var r = min((big.y - big.x) * 1.45, (small.y - small.x) * 1.8 + 0.2);
  r += (laceFbm(uv, 24.0) - 0.5) * 0.2;
  // Places where a strand breaks.
  r = max(r, smoothstep(0.6, 0.78, laceFbm(uv + 0.71, 9.0)) * 0.45);
  r = clamp(r, 0.0, 1.0);
  let bubbles = smoothstep(0.3, 0.05, laceCells(uv + 0.33, 95.0).x) * smoothstep(0.6, 0.08, r);
  return vec4f(r, bubbles, laceFbm(uv + 0.13, 4.0), 1.0);
}
`);
const laceAt = wgslFn(/* wgsl */ `fn swashLace(uv: vec2f) -> vec4f { return laceAt(uv); }`, [laceMath]);

export function createLace(renderer: THREE.WebGPURenderer): THREE.StorageTexture {
  const lace = new THREE.StorageTexture(SIZE, SIZE);
  lace.type = THREE.HalfFloatType; lace.format = THREE.RGBAFormat;
  lace.wrapS = lace.wrapT = THREE.RepeatWrapping;
  lace.minFilter = lace.magFilter = THREE.LinearFilter;
  lace.generateMipmaps = false;
  const kernel = Fn(() => {
    const x = int(instanceIndex.mod(SIZE)), y = int(instanceIndex.div(SIZE));
    textureStore(lace, ivec2(x, y), laceAt(vec2(x, y).add(0.5).div(SIZE)) as unknown as V4).toWriteOnly();
  })().compute(SIZE * SIZE);
  renderer.compute(kernel);
  kernel.dispose();
  return lace;
}

/**
 * Foam cover from a lace sample and the amount of foam (0..1). `soft` is the edge width in lace units, one
 * screen pixel or so, so the strands stay sharp without crawling.
 */
export function foamCover(lace: V4, amount: F, soft: F): F {
  const half = max(pow(amount, 1.4).mul(0.9).mul(lace.z.mul(0.5).add(0.75)).add(lace.y.sub(0.5).mul(0.06)), 0).toVar();
  const mat = smoothstep(half.add(soft), half.sub(soft), lace.x).mul(smoothstep(0, 0.05, amount));
  const bubbles = lace.y.mul(smoothstep(half.add(0.25), half, lace.x)).mul(smoothstep(0.02, 0.2, amount)).mul(0.5);
  return max(mat, bubbles);
}
