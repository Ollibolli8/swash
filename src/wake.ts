import * as THREE from "three/webgpu";
import { Fn, clamp, exp, float, floor, instanceIndex, int, ivec2, length, max, smoothstep, storage, texture, textureStore, uniform, vec2, vec4 } from "three/tsl";
import { SURF } from "./surf";

type F = THREE.Node<"float">;
type V4 = THREE.Node<"vec4">;

/**
 * A hand trailed through the water. The pointer moving over the sea presses the surface down along its path, and the
 * disturbance spreads as rings: the wave equation on its own grid over the simulated strip, 1.25 cm cells, much finer
 * than the surf's. The rings die out in a thin film of water, stop at the water's edge and fade within a couple of
 * seconds. A slow hand leaves a few soft rings, a fast one a wake. The renderer adds their slope to the surface, so
 * the light on the sand ripples with them.
 */

/** Metres a cell. */
const DX = 0.0125;
/** Metres a second: how fast the rings spread, about a ripple's speed on still water. */
const SPEED = 0.55;
/** Seconds a step: two a frame at 60 fps. */
const STEP = 1 / 120;
/** How hard the hand presses, metres a step at full strength. */
const PRESS = 0.0016;
/** Metres: the hand's width in the water. */
const HAND = 0.022;
/** Seconds the rings are run after the hand last moved. */
const AWAKE = 5;

export type Wake = {
  /** (height m, slope x, slope y, 0) on the strip; all zero where nothing has moved. */
  texture: THREE.Texture;
  /** The hand moved from a to b (metres on the strip); strength 0 to 1. */
  stir: (a: THREE.Vector2, b: THREE.Vector2, strength: number) => void;
  /** Run the rings on by `seconds` of real time. */
  update: (seconds: number) => void;
  dispose: () => void;
};

/** `enabled` is false where there is no hovering pointer (a phone): then nothing is allocated or run. */
export function createWake(renderer: THREE.WebGPURenderer, surface: THREE.Texture, enabled: boolean): Wake {
  const map = (w: number, h: number) => {
    const t = new THREE.StorageTexture(w, h);
    t.type = THREE.HalfFloatType; t.format = THREE.RGBAFormat;
    t.wrapS = THREE.RepeatWrapping; t.wrapT = THREE.ClampToEdgeWrapping;
    t.minFilter = t.magFilter = THREE.LinearFilter; t.generateMipmaps = false;
    return t;
  };
  if (!enabled) {
    const none = map(1, 1);
    return { texture: none, stir() {}, update() {}, dispose() { none.dispose(); } };
  }

  const NX = Math.round(SURF.W / DX), NY = Math.round((SURF.Y1 - SURF.Y0) / DX), COUNT = NX * NY;
  const C2 = (SPEED * STEP / DX) ** 2;
  const out = map(NX, NY);
  // (height now, height a step ago), twice.
  const state = [new THREE.StorageBufferAttribute(COUNT, 2), new THREE.StorageBufferAttribute(COUNT, 2)];
  const strokes = Array.from({ length: 4 }, () => uniform(new THREE.Vector4(-99, -99, -99, -99)));
  const strengths = uniform(new THREE.Vector4());

  const kernel = (fromAttr: THREE.StorageBufferAttribute, toAttr: THREE.StorageBufferAttribute) => {
    const from = storage(fromAttr, "vec2", COUNT).toReadOnly(), to = storage(toAttr, "vec2", COUNT);
    return Fn(() => {
      const i = float(instanceIndex.mod(NX)).toVar(), j = float(instanceIndex.div(NX)).toVar();
      const wrap = (v: F) => v.sub(floor(v.div(NX)).mul(NX));
      const at = (ii: F, jj: F) => int(clamp(jj, 0, NY - 1).mul(NX).add(wrap(ii)));
      const c = from.element(instanceIndex).toVar();
      const e = from.element(at(i.add(1), j)).x.toVar(), w = from.element(at(i.sub(1), j)).x.toVar();
      const s = from.element(at(i, j.add(1))).x.toVar(), n = from.element(at(i, j.sub(1))).x.toVar();
      const p = vec2(i.add(0.5).mul(DX), j.add(0.5).mul(DX).add(SURF.Y0)).toVar();
      // The water's depth here, from the surf.
      const depth = (texture(surface, vec2(p.x.div(SURF.W), p.y.sub(SURF.Y0).div(SURF.Y1 - SURF.Y0))).level(float(0)) as unknown as V4).w;
      const open = smoothstep(0.004, 0.03, depth).toVar();
      const damp = open.oneMinus().mul(30).add(1.6).mul(STEP);
      const next = c.x.add(c.x.sub(c.y).mul(float(1).sub(damp))).add(e.add(w).add(s).add(n).sub(c.x.mul(4)).mul(C2)).toVar();
      strokes.forEach((stroke, k) => {
        const a = stroke.xy, ab = stroke.zw.sub(a);
        const along = clamp(p.sub(a).dot(ab).div(max(ab.dot(ab), 1e-8)), 0, 1);
        // Pressed down under the hand and up around it, by as much: no lasting trough follows it, only rings.
        const d2 = length(p.sub(a.add(ab.mul(along)))).div(HAND).pow(2).toVar();
        next.subAssign(float(1).sub(d2).mul(exp(d2.negate())).mul(strengths[(["x", "y", "z", "w"] as const)[k]]).mul(PRESS));
      });
      next.mulAssign(open);
      to.element(instanceIndex).assign(vec2(next, c.x));
      textureStore(out, ivec2(int(i), int(j)), vec4(next, e.sub(w).div(2 * DX), s.sub(n).div(2 * DX), 0)).toWriteOnly();
    })().compute(COUNT);
  };
  const kernels = [kernel(state[0], state[1]), kernel(state[1], state[0])];
  const queue: Array<[THREE.Vector2, THREE.Vector2, number]> = [];
  let parity = 0, carry = 0, idle = AWAKE;

  return {
    texture: out,
    stir(a, b, strength) {
      queue.push([a.clone(), b.clone(), strength]);
      if (queue.length > 8) queue.splice(0, queue.length - 8);
      idle = 0;
    },
    update(seconds) {
      if (idle >= AWAKE) return;
      idle += seconds;
      carry += seconds;
      const n = Math.min(4, Math.floor(carry / STEP));
      carry = Math.min(carry - n * STEP, STEP);
      if (!n) return;
      // The hand's path since the last frame, pressed through each of this frame's steps.
      for (let k = 0; k < strokes.length; k++) {
        const stroke = queue.shift();
        if (stroke) { strokes[k].value.set(stroke[0].x, stroke[0].y, stroke[1].x, stroke[1].y); strengths.value.setComponent(k, stroke[2]); }
        else { strokes[k].value.set(-99, -99, -99, -99); strengths.value.setComponent(k, 0); }
      }
      const list: THREE.ComputeNode[] = [];
      for (let s = 0; s < n; s++) { list.push(kernels[parity]); parity = 1 - parity; }
      renderer.compute(list);
    },
    dispose() { kernels.forEach(k => k.dispose()); out.dispose(); },
  };
}
