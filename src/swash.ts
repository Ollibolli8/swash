import * as THREE from "three/webgpu";
import { uniform } from "three/tsl";
import { createLace } from "./foam";
import { WORLD_M } from "./light";
import { createMaterial } from "./material";
import { createCaustics } from "./optics";
import { createRipples } from "./ripples";
import { BANDS, SURF, createSurf } from "./surf";
import { createWake } from "./wake";
import { guardWebGPU } from "./webgpu-guard";
import { paintInk, setWord } from "./writing";

export { BANDS } from "./surf";

/** Seconds of surf simulated before the first frame, so the beach opens already running. */
const WARM = 7.2;
/** The moment shown when motion is reduced: the last swash drained off the sand (so drawing marks it) and the next
 *  bore halfway in, foaming. */
const STILL_AT = 10;
/** Seconds the hand takes to write a metre of a word. */
const WRITE_PACE = 0.32;

export type SwashOptions = {
  /** Fills its box; the beach is drawn at its size. */
  canvas: HTMLCanvasElement;
  /** The sand's maps: albedo (display sRGB), OpenGL normal and roughness, each a 2 m tile. */
  sand: { color: string; normal: string; roughness: string };
  /** CSS font family for writing in the sand (loaded before `write`). */
  font: string;
  /** One still frame instead of the running sea (reduced motion). Drawing and writing still mark the sand. */
  still?: boolean;
  /** After each frame the running sea draws (`now` from requestAnimationFrame), for a host's frame meter. */
  onFrame?: (now: number) => void;
};

export type Listening = {
  /** For each of BANDS bands along the shore: (bores, swash, foam, swell), each about 0 to 1. */
  bands: Float32Array;
  /** Where each band sits in the view, -1 (left edge) to 1 (right), and how much of it is in view (0 to 1). */
  pan: Float32Array;
  presence: Float32Array;
  /** The hand since the last call: how fast it moved over sand and over water (metres a second), and where (-1 to 1). */
  hand: { sand: number; water: number; pan: number };
  /** Whether the sea is running (not paused, still or hidden). */
  moving: boolean;
};

export type Swash = {
  /** Re-reads the canvas's size. */
  resize: () => void;
  /** A pointer at CSS pixels in the canvas: "hover" moves with no button (rings on the water). */
  pointer: (x: number, y: number, kind: "down" | "move" | "up" | "hover") => void;
  /** A hand writes `text` in the sand where the swash reaches it, clear of `keepClear` (boxes in the canvas's CSS
   *  pixels, such as text laid over the beach). Resolves when the last letter is written. */
  write: (text: string, keepClear?: Array<{ left: number; top: number; right: number; bottom: number }>) => Promise<void>;
  play: (on: boolean) => void;
  setStill: (still: boolean) => void;
  /** What the surf and the hand sound like now (sound.ts). */
  listen: () => Promise<Listening>;
  dispose: () => void;
};

type Stroke = { a: THREE.Vector2; b: THREE.Vector2; push: THREE.Vector2 };

/**
 * The beach, seen straight down: the surf solver (surf.ts), the wind's ripples, the pointer's rings, the caustic pass
 * and the material, run at the solver's fixed step and drawn at most 60 times a second. Rejects where there is no
 * WebGPU adapter, so the host can show its fallback.
 */
export async function createSwash(options: SwashOptions): Promise<Swash> {
  const { canvas } = options;
  if (!("gpu" in navigator)) throw new Error("WebGPU unavailable");
  guardWebGPU();
  const params = new URLSearchParams(location.search);
  const debug = params.has("swash-debug");
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, alpha: false, powerPreference: "high-performance" });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  let lost = false, disposed = false;
  renderer.onDeviceLost = () => { lost = true; canvas.dispatchEvent(new Event("swash:lost")); };
  try {
    await renderer.init();
    if (!("isWebGPUBackend" in renderer.backend)) throw new Error("WebGPU adapter unavailable");
  } catch (error) { renderer.dispose(); throw error; }

  const cleanup: Array<() => void> = [() => renderer.dispose()];
  const release = () => { for (const step of cleanup.toReversed()) step(); cleanup.length = 0; };
  try {
    const coarse = matchMedia("(pointer: coarse)").matches;
    // Phones get 5 cm cells, desktops 3.75 cm.
    const surf = createSurf(renderer, !coarse);
    cleanup.push(() => surf.dispose());
    const u = { time: uniform(0), aspect: uniform(1), cam: uniform(-0.46) };
    const loader = new THREE.TextureLoader();
    const [sand, normal, roughness] = await Promise.all([options.sand.color, options.sand.normal, options.sand.roughness].map(url =>
      loader.loadAsync(url).then(t => {
        cleanup.push(() => t.dispose());
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.colorSpace = THREE.NoColorSpace;
        t.anisotropy = Math.min(4, renderer.getMaxAnisotropy());
        return t;
      })));
    const ripples = createRipples(renderer);
    cleanup.push(() => ripples.dispose());
    // Rings from a pointer moving over the water (none on touch screens, where nothing hovers).
    const wake = createWake(renderer, surf.surface, !coarse);
    cleanup.push(() => wake.dispose());
    const optics = createCaustics(renderer, u, surf.surface, ripples.texture, wake.texture);
    cleanup.push(() => optics.dispose());
    const lace = createLace(renderer);
    cleanup.push(() => lace.dispose());
    const material = createMaterial({ ...u, sand, normal, roughness, caustics: optics.texture, lace, ripples: ripples.texture, wake: wake.texture,
      surface: surf.surface, drift: surf.drift, matter: surf.matter, groove: surf.groove, cells: [surf.nx, surf.ny], debugView: debug && params.has("surf-view") });
    cleanup.push(() => material.dispose());
    const quad = new THREE.QuadMesh(material);

    let width = 1, height = 1, still = !!options.still, playing = false, visible = !document.hidden;
    let raf = 0, redrawRaf = 0, previous = 0, accumulator = 0, slow = 0;
    let last: THREE.Vector2 | null = null, lastAt = 0, down = false, hovered: THREE.Vector2 | null = null, hoveredAt = 0;
    const strokes: Stroke[] = [];
    let pixelRatio = Math.min(devicePixelRatio || 1, coarse ? 1.25 : 1.75);
    // At most this many solver steps a frame; a slower device lets the sea run slow rather than fall behind.
    const maxSteps = Math.ceil(1 / 30 / surf.dt);
    // The writing hand: where it is along the word, and when it gets there.
    let hand: { from: number; to: number; seconds: number; t: number; done: () => void } | null = null;
    const heard = { sand: 0, water: 0, x: 0 };

    const draw = () => {
      if (disposed || lost || !visible) return;
      u.time.value = surf.time();
      ripples.update(u.time.value);
      optics.update();
      quad.render(renderer);
    };
    /** Screen-height world units to metres on the strip, wrapped into it. */
    const toStrip = (p: THREE.Vector2) => {
      const x = (p.x - u.aspect.value / 2) * WORLD_M + SURF.W / 2;
      return new THREE.Vector2(x - Math.floor(x / SURF.W) * SURF.W, p.y * WORLD_M);
    };
    const moveHand = (delta: number) => {
      if (!hand) return;
      hand.t = Math.min(hand.seconds, hand.t + delta);
      const k = hand.t / hand.seconds, eased = k * k * (3 - 2 * k) * 0.35 + k * 0.65;
      surf.inkTo.value = hand.from + (hand.to - hand.from) * eased;
      surf.inkOn.value = 1;
      heard.sand = Math.max(heard.sand, (hand.to - hand.from) / hand.seconds * 0.8);
      heard.x = ((hand.from + hand.to) / 2 - SURF.W / 2) / (u.aspect.value * WORLD_M / 2);
      if (hand.t >= hand.seconds) { const { done } = hand; hand = null; queueMicrotask(done); }
    };
    const step = (n: number) => {
      surf.brushRadius.value = coarse ? 0.08 : 0.06;
      const push = new THREE.Vector2();
      for (let i = 0; i < surf.brushes.length; i++) {
        const stroke = strokes.shift();
        if (stroke) { surf.brushes[i].value.set(stroke.a.x, stroke.a.y, stroke.b.x, stroke.b.y); push.add(stroke.push); }
        else surf.brushes[i].value.set(-99, -99, -99, -99);
      }
      surf.push.value.copy(push.clampLength(0, 2.4));
      surf.run(n);
      // The ink is pressed in while the hand writes, and once more after it lifts.
      if (!hand) surf.inkOn.value = 0;
    };
    const advance = (seconds: number) => {
      let left = Math.round(seconds / surf.dt);
      while (left > 0) { const n = Math.min(left, 120); step(n); left -= n; }
    };
    const redraw = () => {
      if (playing && !still || redrawRaf || disposed || !visible) return;
      redrawRaf = requestAnimationFrame(() => { redrawRaf = 0; draw(); });
    };
    const frame = (now: number) => {
      if (disposed || lost || !running()) return;
      raf = requestAnimationFrame(frame);
      if (previous && now - previous < 1000 / 62) return;
      const elapsed = previous ? (now - previous) / 1000 : 1 / 60;
      slow = elapsed > 0.025 ? slow + 1 : Math.max(0, slow - 1);
      if (slow > 45 && pixelRatio > 0.8) { pixelRatio *= 0.85; renderer.setPixelRatio(pixelRatio); renderer.setSize(width, height, false); slow = 0; }
      previous = now;
      const delta = Math.min(0.05, elapsed);
      moveHand(delta);
      accumulator += delta;
      const n = Math.min(maxSteps, Math.floor(accumulator / surf.dt));
      accumulator = Math.min(accumulator - n * surf.dt, surf.dt);
      if (n > 0) step(n);
      wake.update(delta);
      draw();
      options.onFrame?.(now);
    };
    const running = () => playing && !still && visible;
    const sync = () => {
      cancelAnimationFrame(raf); previous = 0;
      if (running() && !disposed) raf = requestAnimationFrame(frame);
      else redraw();
      canvas.dataset.playing = String(running());
    };
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = Math.max(1, rect.width); height = Math.max(1, rect.height);
      u.aspect.value = width / height;
      // The swash zone a little above the middle (lower on a tall screen): dry sand above it to write on, the surf below.
      u.cam.value = width >= height ? -0.46 : -0.52;
      renderer.setPixelRatio(pixelRatio); renderer.setSize(width, height, false);
      optics.resize(width, height);
      redraw();
    };
    const onVisibility = () => { visible = !document.hidden; sync(); };
    document.addEventListener("visibilitychange", onVisibility);
    cleanup.push(() => document.removeEventListener("visibilitychange", onVisibility));

    canvas.dataset.backend = "webgpu"; canvas.dataset.sim = `swe-${surf.nx}x${surf.ny}`;
    resize();
    advance(still ? STILL_AT : WARM);
    draw();
    if (lost) throw new Error("WebGPU device lost while starting");

    const swash: Swash = {
      resize,
      pointer(x, y, kind) {
        if (kind === "up") { down = false; last = null; return; }
        const p = toStrip(new THREE.Vector2(x / height, u.cam.value + y / height));
        const now = performance.now();
        const near = hovered && (now - hoveredAt < 150 || p.distanceTo(hovered) < 0.25) && Math.abs(p.x - hovered.x) < SURF.W / 2;
        const speed = hovered && near ? p.distanceTo(hovered) / Math.max(1 / 240, (now - hoveredAt) / 1000) : 0;
        // A hand trailed through the water: rings, stronger the faster it moves (none while the sea is still).
        if (running() && (kind === "hover" || down) && hovered && near) wake.stir(hovered, p, Math.min(1, 0.25 + speed / 1.2));
        if (hovered && near && (kind === "hover" || down)) {
          const pan = x / width * 2 - 1;
          // Over water, or digging the sand (a hover over the sand makes no sound).
          if (p.y > -0.15) { heard.water = Math.max(heard.water, speed); heard.x = pan; }
          else if (down) { heard.sand = Math.max(heard.sand, speed); heard.x = pan; }
        }
        hovered = p; hoveredAt = now;
        if (kind === "hover") return;
        if (kind === "down") { down = true; last = p.clone(); lastAt = now; }
        if (down && last) {
          // A stroke across the strip's seam would span it; start afresh on the other side.
          if (Math.abs(p.x - last.x) > SURF.W / 2) last.copy(p);
          const seconds = Math.max(1 / 120, (now - lastAt) / 1000);
          if (p.y >= SURF.Y0 && p.y <= SURF.Y1) strokes.push({ a: last.clone(), b: p.clone(), push: p.clone().sub(last).divideScalar(seconds).clampLength(0, 2.4) });
          if (strokes.length > 48) strokes.splice(0, strokes.length - 48);
          last.copy(p); lastAt = now;
          if (!running()) { step(1); surf.settle(); draw(); }
        }
      },
      write(text, keepClear = []) {
        hand?.done();
        hand = null;
        // About two thirds of the view wide (more on a phone), and never so wide that the strip's next copy shows it.
        const view = u.aspect.value * WORLD_M, portrait = u.aspect.value < 1;
        const maxWidth = Math.min(view * (portrait ? 0.84 : 0.62), 2 * SURF.W - view - 0.6, 6.2);
        let word = setWord(text, options.font, 1.5, maxWidth, SURF.W / 2, -0.75, 0.022);
        // Clear of what lies over the beach, in order of preference: moved along the shore, set smaller beside it,
        // set smaller below it above the same line, moved down the beach.
        const pxPerM = height / WORLD_M, margin = 0.15;
        const toPy = (y: number) => (y / WORLD_M - u.cam.value) * height;
        /** Metres from the view's middle; the word's copy one strip along stays out of view. */
        const rightLimit = Math.min(width / 2 / pxPerM, SURF.W - view / 2) - margin;
        for (const box of keepClear) {
          if (!word) break;
          const left = (word.x - SURF.W / 2 - margin) * pxPerM + width / 2, right = left + (word.width + 2 * margin) * pxPerM;
          const top = toPy(word.y - word.ascent - margin), bottom = toPy(word.y + word.descent + margin);
          if (right < box.left || left > box.right || bottom < box.top || top > box.bottom) continue;
          const from = (box.right - width / 2) / pxPerM + margin, room = rightLimit - from;
          const below = word.y - margin - (box.bottom / height + u.cam.value) * WORLD_M;
          if (room >= word.width * 0.55) {
            const fit = Math.min(word.width, room);
            word = setWord(text, options.font, word.size * fit / word.width, fit, SURF.W / 2 + from + fit / 2, word.y, 0.022);
          } else if (below >= word.ascent * 0.55) {
            word = setWord(text, options.font, word.size * Math.min(1, below / word.ascent), maxWidth, SURF.W / 2, word.y, 0.022);
          } else word = { ...word, y: word.y + (box.bottom - top) / pxPerM };
        }
        const span = paintInk(surf.ink.image as HTMLCanvasElement, surf.inkScale, SURF.Y0, word);
        surf.ink.needsUpdate = true;
        if (!word || !span) return Promise.resolve();
        // The hand smooths the sand flat where the word goes and where the last one was (if the sea has left any of
        // it), passing over both, and writes as it goes.
        surf.inkWas.value.copy(surf.inkBox.value);
        surf.inkBox.value.set(word.x - 2 * word.radius, word.y - word.ascent - 2 * word.radius,
          word.x + word.width + 2 * word.radius, word.y + word.descent + 2 * word.radius);
        const from = Math.min(span.from, surf.inkWas.value.x > -90 ? surf.inkWas.value.x : span.from);
        const to = Math.max(span.to, surf.inkWas.value.z);
        return new Promise<void>(done => {
          hand = { from, to, seconds: (to - from) * WRITE_PACE + 0.3, t: 0, done };
          if (!running()) { hand.t = hand.seconds; moveHand(0); step(1); surf.settle(); draw(); }
        });
      },
      play(on) { playing = on; sync(); },
      setStill(on) { still = on; sync(); },
      async listen() {
        const bands = await surf.measure();
        // The strip's bands, from its middle (the view's middle) out to either side, wrapped to the nearest copy.
        const halfView = u.aspect.value * WORLD_M / 2;
        const pan = new Float32Array(BANDS), presence = new Float32Array(BANDS);
        for (let b = 0; b < BANDS; b++) {
          let x = (b + 0.5) / BANDS * SURF.W - SURF.W / 2;
          x -= Math.round(x / SURF.W) * SURF.W;
          pan[b] = Math.max(-1, Math.min(1, x / halfView));
          presence[b] = Math.exp(-Math.max(0, Math.abs(x) - halfView) / 1.5);
        }
        const heardNow = { sand: heard.sand, water: heard.water, pan: Math.max(-1, Math.min(1, heard.x)) };
        heard.sand *= 0.5; heard.water *= 0.5;
        return { bands, pan, presence, hand: heardNow, moving: running() };
      },
      dispose() {
        if (disposed) return;
        disposed = true;
        cancelAnimationFrame(raf); cancelAnimationFrame(redrawRaf);
        hand?.done(); hand = null;
        release();
        if (debug) delete (window as unknown as { __swash?: unknown }).__swash;
      },
    };
    if (debug) (window as unknown as { __swash?: unknown }).__swash = {
      ...swash,
      /** Restart the surf so that `t` seconds have run. */
      time(t: number) { surf.reset(0); advance(Math.max(0, t)); draw(); },
      advance(seconds: number) { moveHand(seconds); advance(Math.min(60, Math.max(0, seconds))); draw(); },
    };
    return swash;
  } catch (error) {
    disposed = true;
    release();
    throw error;
  }
}
