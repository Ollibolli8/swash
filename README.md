# Swash

Write in the sand and the sea takes it back. A live simulation of the surf on a beach, seen from straight above:
each wave breaks, runs up the sand, stops and drains away, and fills whatever you draw or write in its path. Draw
with a finger or the mouse, or type a word and a hand writes it.

**[Live demo](https://ollibolli8.github.io/swash/)** · Maquette 02 on [work.draindk.tech](https://work.draindk.tech/lab/swash/)

WebGPU compute and three.js TSL, in TypeScript. The same surf runs under a client's website.

## How it works

**The surf** (`src/surf.ts`) is the shallow-water equations on the GPU, the way coastal engineers model a swash
zone, over a strip of beach 9.6 m along the shore and 6 m from dry sand out to sea: 3.75 cm cells (256 × 160) and
300 steps a second, or 5 cm and 225 on a phone. Staggered grid, semi-Lagrangian momentum, upwind mass-conserving
continuity with hydrostatic face depths (so wet/dry fronts move without leaks), Manning friction, and a relaxation
zone at sea that feeds in the waves and absorbs what the beach reflects. The strip is periodic along the shore, so
any screen width tiles it. The beach is a 1:7 face with low cusps, a 1:4 terrace where waves run in as bores, and a
step down to deep water.

- **Breaking and foam:** a steep, rising front over the shallows breaks. Foam is made there, conserved on the
  surface (it piles up where the water converges, thins where it spreads), bursts from a mat into lace within a
  second, is left on the sand when the water drains and pops there. The lace is Dan Greenheck's idea from Tidewater:
  store the distance to the nearest bubble strand and threshold it by how much foam there is.
- **Sand in the water:** stirred up by the bores and fast thin swash, it clouds the water and settles; the sand
  stays dark behind the water and dries over about twenty seconds.

**Drawing** digs a channel in the solver's cells, so the next swash runs into it and down it. What you see is a
field four times finer (`groove` in `src/surf.ts`): a rounded trench with the sand it pushed aside heaped along its
edges. Under moving water both slump and fill at `wet × (0.6 v² + 0.03)` per second, after the rule that a sheet
flow carries sand roughly as the square of its speed: a still pool barely touches a letter, a running swash wipes
it out in a few waves.

**Writing** (`src/writing.ts`): the word is set in metres, kept clear of any text over the beach, and painted into
a canvas over the strip as a distance field: 24 stacked fills and strokes of the text, widest and faintest first,
which canvas antialiasing blends into a smooth ramp with no distance transform. The groove reads it at full
resolution, the solver through the mip whose pixels are its cells' size. A hand reveals it left to right and
smooths the sand flat just ahead of itself, over the last word too.

**Light** (`src/light.ts`, `src/optics.ts`, `src/material.ts`): the sun of a July afternoon at 55° N. Sunlight is
refracted through the simulated surface onto the bed with Evan Wallace's projected-mesh caustics, so a bore's front
and the swash's edge focus into bright lines on the sand. The wind's centimetre ripples (`src/ripples.ts`) are a
spectrum summed every frame at their true speeds, capillary waves included; the shortest only glint, since their
light comes to a focus within a few centimetres of water. Red and blue land a little apart in the sharpest folds.
Clear coastal water absorbs, the stirred sand scatters, the sky reflects with Fresnel. Moving the mouse over the
water trails rings (`src/wake.ts`, the wave equation on its own 1.25 cm grid).

**Sound** (`src/sound.ts`, off until turned on): synthesized from the simulation, nothing sampled. A compute pass
measures, in eight bands along the shore, the breaking bores, the swash running over the sand, the foam bursting
and the water's motion; each drives its own filtered noise (the foam a crackle of tiny rings), panned to where it
happens. A finger in the sand scrapes, a hand in the water swishes.

Capped at 60 fps; a device that can't keep up gets fewer pixels. One still frame when motion is reduced (drawing
and writing still mark the sand). Without WebGPU, `createSwash` rejects and the demo shows a recording.

## Use it

```ts
import { createSwash } from "./src/swash";
import { createSound } from "./src/sound";

const font = '"Nothing You Could Do"';
await document.fonts.load(`100px ${font}`);       // the hand writes in it
const swash = await createSwash({                  // rejects without WebGPU
  canvas: document.querySelector("canvas"),        // fills its box
  font,
  sand: { color: "sand/sand_03_diff_1k.jpg", normal: "sand/sand_03_nor_gl_1k.jpg", roughness: "sand/sand_03_rough_1k.jpg" },
  still: matchMedia("(prefers-reduced-motion: reduce)").matches,
});
swash.play(true);
swash.pointer(x, y, "down");                       // CSS pixels in the canvas; "move", "up", or "hover" over water
await swash.write("Swash", [headerRect]);          // resolves when the hand lifts
const sound = createSound(swash);
sound.set(true);                                   // from a click
```

The canvas fires `swash:lost` if the GPU device is lost. `demo/main.ts` wires pointer events, the form and the
fallback.

## Build the demo

```sh
npm install
npm run build   # docs/ (GitHub Pages)
npm run dev     # build and serve
```

## Credits

- Sand: [Sand 03](https://polyhaven.com/a/sand_03) by Charlotte Baglioni, Poly Haven, CC0.
- Handwriting: [Nothing You Could Do](https://fonts.google.com/specimen/Nothing+You+Could+Do) by Kimberly Geswein,
  SIL Open Font License 1.1, through Fontsource.
- Caustics after Evan Wallace, [Rendering Realtime Caustics in WebGL](https://medium.com/@evanwallace/rendering-realtime-caustics-in-webgl-2a99a29a0b2c)
  (reference implementation MIT). Foam lace after Dan Greenheck's Tidewater (MIT). Solver after Chentanez & Müller,
  *Real-time Simulation of Large Bodies of Water with Small Scale Details* (2010), and Bridson, *Fluid Simulation for
  Computer Graphics*.

## Licence

MIT, Oliver Kure Olsen, 2026. The sand maps and the font keep their own licences above.
