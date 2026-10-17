/**
 * three r186 hands every texture view `swizzle: "rgba"`, the current spec's string. Chromium builds that still type
 * that member as the older GPUTextureComponentSwizzle dictionary (Chrome 141 among them) reject the string, so every
 * createView throws and a three WebGPU scene dies on its first pass. "rgba" is the identity, the same as no swizzle,
 * so the first time a view is refused for it this drops the member from then on. Browsers that take the string never
 * reach the catch. Called before the renderer starts; calling it twice is harmless.
 *
 * Its compileAsync also gives the page a turn after every object and every shader stage it builds: scheduler.yield()
 * where there is one, otherwise a requestAnimationFrame, a whole frame a turn. Safari has no scheduler, so a compile
 * there took a frame a stage, and Chrome runs yield's continuations ahead of rendering, so a long compile drew a frame
 * only every 100 ms or so, when its starvation guard let one through. So the turn is a message everywhere: a task like
 * any other, so a frame that is due goes first.
 */
export function guardWebGPU() {
  paceYield();
  type Desc = { swizzle?: unknown; [key: string]: unknown };
  type CreateView = ((this: object, d?: Desc) => unknown) & { guarded?: true };
  const proto = (globalThis as { GPUTexture?: { prototype: { createView: CreateView } } }).GPUTexture?.prototype;
  if (!proto || proto.createView.guarded) return;
  const createView = proto.createView;
  let strip = false;
  const guarded: CreateView = function (this: object, d?: Desc) {
    if (strip && d?.swizzle === "rgba") { const { swizzle: _, ...rest } = d; return createView.call(this, rest); }
    try { return createView.call(this, d); } catch (e) {
      if (strip || !(e instanceof TypeError) || d?.swizzle !== "rgba") throw e;
      strip = true;
      const { swizzle: _, ...rest } = d;
      return createView.call(this, rest);
    }
  };
  guarded.guarded = true;
  proto.createView = guarded;
}

function paceYield() {
  type Turn = (() => Promise<void>) & { paced?: true };
  const g = globalThis as { scheduler?: { yield?: Turn } };
  if (g.scheduler?.yield?.paced || typeof MessageChannel === "undefined") return;
  const waiting: (() => void)[] = [], channel = new MessageChannel();
  channel.port1.onmessage = () => waiting.shift()?.();
  const turn: Turn = () => new Promise<void>(r => { waiting.push(r); channel.port2.postMessage(0); });
  turn.paced = true;
  if (g.scheduler) g.scheduler.yield = turn;
  else Object.defineProperty(g, "scheduler", { value: { yield: turn }, configurable: true, writable: true });
}
