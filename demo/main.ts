import "@fontsource/nothing-you-could-do";
import "./style.css";
import { createSwash, type Swash } from "../src/swash";
import { createSound } from "../src/sound";

const host = document.getElementById("beach")!;
const canvas = host.querySelector<HTMLCanvasElement>(".sw-beach")!;
const form = host.querySelector<HTMLFormElement>(".sw-write")!;
const input = form.querySelector("input")!;
const soundButton = host.querySelector<HTMLButtonElement>(".sw-sound")!;
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const FONT = '"Nothing You Could Do"';

/** No WebGPU: a recording of the beach in place of the live one, filmed on a wide screen or a phone. */
function fallBack() {
  host.dataset.gpu = "off";
  host.querySelector<HTMLElement>(".sw-note")!.hidden = false;
  soundButton.hidden = true;
  const tall = matchMedia("(orientation: portrait)").matches ? "-tall" : "";
  const film = document.createElement("video");
  film.className = "sw-film";
  // H.264 where the browser has it, else VP9 (a Chromium built without proprietary codecs).
  const type = film.canPlayType('video/mp4; codecs="avc1.640028"') ? "mp4" : "webm";
  Object.assign(film, { src: `swash${tall}.${type}`, poster: `poster${tall}.webp`, muted: true, loop: true, playsInline: true, autoplay: !reduced.matches });
  film.setAttribute("aria-hidden", "true");
  host.querySelector("picture")!.after(film);
}

async function start() {
  let swash: Swash;
  try {
    // The word is written in the font, so it has to be here first.
    await document.fonts.load(`100px ${FONT}`);
    swash = await createSwash({
      canvas, font: FONT, still: reduced.matches,
      sand: { color: "sand/sand_03_diff_1k.jpg", normal: "sand/sand_03_nor_gl_1k.jpg", roughness: "sand/sand_03_rough_1k.jpg" },
    });
  } catch {
    fallBack();
    return;
  }
  host.dataset.gpu = "on";
  canvas.addEventListener("swash:lost", () => { swash.dispose(); sound.dispose(); fallBack(); }, { once: true });
  swash.play(true);
  reduced.addEventListener("change", e => swash.setStill(e.matches));
  new ResizeObserver(() => swash.resize()).observe(canvas);

  // Drawing: a finger or a button held down on the beach digs; a mouse moving over the water ruffles it.
  let drawing: number | null = null;
  const at = (e: PointerEvent): [number, number] => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  canvas.addEventListener("pointerdown", e => {
    if (drawing !== null || e.button > 0) return;
    drawing = e.pointerId;
    canvas.setPointerCapture(e.pointerId);
    swash.pointer(...at(e), "down");
  });
  canvas.addEventListener("pointermove", e => {
    if (e.pointerId === drawing) for (const c of e.getCoalescedEvents?.() ?? [e]) swash.pointer(...at(c), "move");
    else if (drawing === null && e.pointerType === "mouse") swash.pointer(...at(e), "hover");
  });
  const lift = (e: PointerEvent) => { if (e.pointerId === drawing) { drawing = null; swash.pointer(...at(e), "up"); } };
  canvas.addEventListener("pointerup", lift);
  canvas.addEventListener("pointercancel", lift);

  // The word keeps clear of the text over the beach.
  const head = host.querySelector<HTMLElement>(".sw-head")!;
  const clear = () => {
    const c = canvas.getBoundingClientRect(), h = head.getBoundingClientRect();
    return [{ left: h.left - c.left, top: h.top - c.top, right: h.right - c.left, bottom: h.bottom - c.top }];
  };
  form.addEventListener("submit", e => {
    e.preventDefault();
    const word = input.value.trim();
    if (!word) { input.focus(); return; }
    void swash.write(word, clear());
    input.value = "";
    // On a phone the keyboard would hide the beach.
    if (matchMedia("(pointer: coarse)").matches) input.blur();
  });

  const sound = createSound(swash);
  soundButton.addEventListener("click", () => {
    const on = !sound.on;
    sound.set(on);
    soundButton.setAttribute("aria-pressed", String(on));
    soundButton.textContent = on ? "Sound on" : "Sound off";
  });

  // The name, written by the hand as the page opens.
  setTimeout(() => void swash.write("Swash", clear()), reduced.matches ? 0 : 700);
}

void start();
