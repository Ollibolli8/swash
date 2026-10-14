/**
 * Writing in the sand: a word laid out in metres on the beach and painted into the solver's ink canvas (surf.ts) as a
 * distance field, so the sand can be pressed into a rounded trench with a heap of sand along it, like a finger's.
 *
 * The green channel falls from 1 on the letters' strokes to 0 two radii away. It is painted as stacked strokes of
 * the text, the widest and faintest first, each narrower one brighter: canvas antialiasing blends one step into the
 * next, so the field comes out smooth without a distance transform.
 */

const LAYERS = 24;
const measure = document.createElement("canvas").getContext("2d")!;

export type Lettering = {
  word: string;
  /** CSS font family; it must be loaded. */
  font: string;
  /** Metres: the em size. */
  size: number;
  /** Metres on the strip: the word's left edge along the shore, and its baseline (y grows seaward). */
  x: number;
  y: number;
  /** Metres: the word's width, and how far it reaches above and below the baseline. */
  width: number;
  ascent: number;
  descent: number;
  /** Metres: the trench's radius beyond the letters' own strokes. */
  radius: number;
};

/** Sets `text` at `size` metres to the em (smaller if it would be wider than `maxWidth`), centred on x, with a
 *  trench `stroke` ems wide either side of the letters' own strokes. */
export function setWord(text: string, font: string, size: number, maxWidth: number, x: number, y: number, stroke: number): Lettering | null {
  const word = text.replace(/\s+/g, " ").trim();
  if (!word) return null;
  measure.font = `100px ${font}`;
  const box = measure.measureText(word);
  const em = (box.actualBoundingBoxLeft + box.actualBoundingBoxRight) / 100;
  size = Math.min(size, maxWidth / em);
  const width = em * size;
  return {
    word, font, size, x: x - width / 2, y, width,
    ascent: box.actualBoundingBoxAscent / 100 * size, descent: box.actualBoundingBoxDescent / 100 * size, radius: stroke * size,
  };
}

/**
 * Paints the word into the ink canvas at `scale` pixels a metre, whose top row is the strip's landward edge `y0`
 * (no word clears it). Returns where the written strokes start and end along the shore, metres, for the hand.
 */
export function paintInk(canvas: HTMLCanvasElement, scale: number, y0: number, w: Lettering | null): { from: number; to: number } | null {
  const g = canvas.getContext("2d")!;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.globalCompositeOperation = "source-over";
  g.fillStyle = "#000";
  g.fillRect(0, 0, canvas.width, canvas.height);
  if (!w) return null;
  g.font = `${w.size * scale}px ${w.font}`;
  g.textAlign = "left"; g.textBaseline = "alphabetic";
  g.lineJoin = "round"; g.lineCap = "round";
  const px = w.x * scale + g.measureText(w.word).actualBoundingBoxLeft, py = (w.y - y0) * scale;
  for (let k = 0; k < LAYERS; k++) {
    const v = Math.round(255 * (k + 1) / LAYERS);
    g.fillStyle = g.strokeStyle = `rgb(0 ${v} 0)`;
    // Two radii out for the widest, down to the letter itself for the brightest.
    const reach = 2 * w.radius * (1 - k / LAYERS) * scale;
    g.fillText(w.word, px, py);
    if (reach > 0.5) { g.lineWidth = reach * 2; g.strokeText(w.word, px, py); }
  }
  return { from: w.x - w.radius, to: w.x + w.width + w.radius };
}
