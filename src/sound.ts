import { BANDS, type Swash } from "./swash";

/**
 * The surf's sound, synthesized from the simulation itself: nothing is sampled or looped from a recording. Each
 * frame or two the solver measures, in bands along the shore (surf.ts), the four things that make the sound of a
 * beach, and each drives its own noise:
 *
 * - bores: the roar of broken waves running in, band-passed noise that brightens as they get louder;
 * - swash: the hiss of the thin sheet running up and over the sand;
 * - foam: its bubbles bursting, a crackle of tiny clicks;
 * - swell: the low wash of the water moving under it all.
 *
 * Each is panned by where it happens in the view. A finger digging the sand scrapes, a hand trailed through the
 * water swishes. Off until turned on (browsers start sound only from a click), quiet while the tab is hidden.
 */

export type Sound = { readonly on: boolean; set: (on: boolean) => void; dispose: () => void };

type Voice = { gain: GainNode; filter: BiquadFilterNode };

export function createSound(swash: Swash): Sound {
  let ctx: AudioContext | null = null, on = false, polling = false, disposed = false;
  let master: GainNode;
  let layers: Array<{ left: Voice; right: Voice }> = [];
  let scrape: { gain: GainNode; pan: StereoPannerNode }, swish: { gain: GainNode; pan: StereoPannerNode };

  /** Seconds of white noise, and of a crackle: clicks of random size at random times, like bubbles bursting. */
  const buffers = (audio: AudioContext) => {
    const rate = audio.sampleRate, length = rate * 4;
    const white = audio.createBuffer(1, length, rate), crackle = audio.createBuffer(1, length, rate);
    const w = white.getChannelData(0), c = crackle.getChannelData(0);
    for (let i = 0; i < length; i++) w[i] = Math.random() * 2 - 1;
    for (let t = 0; t < length;) {
      t += Math.ceil(-Math.log(1 - Math.random()) * rate / 420);
      // Each a few cycles of a ring at a few kilohertz, dying in a millisecond or two.
      const size = -Math.log(1 - Math.random()) * 0.5, decay = rate * (0.0003 + Math.random() * 0.0012);
      const w = 2 * Math.PI * (1800 + Math.random() * 5200) / rate, phase = Math.random() * 6.283;
      for (let k = 0; k < decay * 4 && t + k < length; k++) c[t + k] += size * Math.exp(-k / decay) * Math.sin(w * k + phase);
    }
    return { white, crackle };
  };

  const build = (audio: AudioContext) => {
    const { white, crackle } = buffers(audio);
    master = audio.createGain();
    master.gain.value = 0;
    const limiter = audio.createDynamicsCompressor();
    limiter.threshold.value = -16; limiter.ratio.value = 4; limiter.attack.value = 0.005; limiter.release.value = 0.25;
    master.connect(limiter).connect(audio.destination);
    /** A noise through a filter to a gain, panned; each started at a different point so left and right differ. */
    const voice = (buffer: AudioBuffer, type: BiquadFilterType, frequency: number, q: number, pan: number, extra?: BiquadFilterNode) => {
      const source = audio.createBufferSource();
      source.buffer = buffer; source.loop = true;
      const filter = audio.createBiquadFilter();
      filter.type = type; filter.frequency.value = frequency; filter.Q.value = q;
      const gain = audio.createGain();
      gain.gain.value = 0;
      const panner = audio.createStereoPanner();
      panner.pan.value = pan;
      let chain: AudioNode = source.connect(filter);
      if (extra) chain = chain.connect(extra);
      chain.connect(gain).connect(panner).connect(master);
      source.start(0, Math.random() * buffer.duration);
      return { gain, filter, panner };
    };
    const pair = (buffer: AudioBuffer, type: BiquadFilterType, frequency: number, q: number, extra: () => BiquadFilterNode | undefined) => ({
      left: voice(buffer, type, frequency, q, -0.85, extra()), right: voice(buffer, type, frequency, q, 0.85, extra()),
    });
    const highpass = (f: number) => () => { const h = audio.createBiquadFilter(); h.type = "highpass"; h.frequency.value = f; h.Q.value = 0.5; return h; };
    const lowpass = (f: number) => () => { const l = audio.createBiquadFilter(); l.type = "lowpass"; l.frequency.value = f; l.Q.value = 0.5; return l; };
    layers = [
      pair(white, "lowpass", 900, 0.4, highpass(110)),    // bores
      pair(white, "highpass", 2400, 0.5, lowpass(9000)),  // swash
      pair(crackle, "highpass", 2200, 0.6, lowpass(11000)), // foam
      pair(white, "lowpass", 240, 0.5, highpass(35)),     // swell
    ];
    const scrapeVoice = voice(white, "bandpass", 1600, 1.1, 0);
    const swishVoice = voice(white, "bandpass", 520, 0.8, 0, lowpass(1400)());
    scrape = { gain: scrapeVoice.gain, pan: scrapeVoice.panner };
    swish = { gain: swishVoice.gain, pan: swishVoice.panner };
  };

  /** Loudness of each layer at full measure, and how fast it follows (seconds). */
  const LEVEL = [0.55, 0.22, 0.3, 0.32], FOLLOW = [0.09, 0.06, 0.12, 0.25];

  const poll = async () => {
    if (polling) return;
    polling = true;
    while (on && !disposed && ctx) {
      const started = performance.now();
      try {
        const heard = await swash.listen();
        if (!on || disposed || !ctx) break;
        const now = ctx.currentTime;
        for (let layer = 0; layer < 4; layer++) {
          let left = 0, right = 0;
          for (let b = 0; b < BANDS; b++) {
            const v = heard.bands[b * 4 + layer] * heard.presence[b], a = (heard.pan[b] + 1) * Math.PI / 4;
            left += v * Math.cos(a); right += v * Math.sin(a);
          }
          // The surf is quieter when the sea stands still.
          const k = LEVEL[layer] * (heard.moving ? 1 : 0.25) * 2 / BANDS;
          const { left: l, right: r } = layers[layer];
          // The swell keeps a quiet floor: the sea is never silent.
          const floor = layer === 3 ? 0.04 : 0;
          l.gain.gain.setTargetAtTime(floor + Math.pow(left * k, 0.85), now, FOLLOW[layer]);
          r.gain.gain.setTargetAtTime(floor + Math.pow(right * k, 0.85), now, FOLLOW[layer]);
          if (layer === 0) {
            // A louder roar is a brighter one.
            const bright = 700 + Math.min(1, (left + right) / BANDS) * 2200;
            l.filter.frequency.setTargetAtTime(bright, now, 0.15); r.filter.frequency.setTargetAtTime(bright, now, 0.15);
          }
        }
        scrape.gain.gain.setTargetAtTime(Math.min(1, heard.hand.sand / 1.6) * 0.32, now, 0.03);
        scrape.pan.pan.setTargetAtTime(heard.hand.pan * 0.8, now, 0.05);
        swish.gain.gain.setTargetAtTime(Math.min(1, heard.hand.water / 1.4) * 0.3, now, 0.05);
        swish.pan.pan.setTargetAtTime(heard.hand.pan * 0.8, now, 0.05);
      } catch { break; }
      // About thirty times a second.
      await new Promise(r => setTimeout(r, Math.max(0, 33 - (performance.now() - started))));
    }
    polling = false;
  };

  const onVisibility = () => {
    if (!ctx || !on) return;
    if (document.hidden) void ctx.suspend(); else { void ctx.resume(); void poll(); }
  };
  document.addEventListener("visibilitychange", onVisibility);

  return {
    get on() { return on; },
    set(next) {
      if (disposed || next === on) return;
      on = next;
      if (on) {
        if (!ctx) { ctx = new AudioContext(); build(ctx); }
        void ctx.resume();
        master.gain.setTargetAtTime(0.9, ctx.currentTime, 0.3);
        void poll();
      } else if (ctx) {
        const audio = ctx;
        master.gain.setTargetAtTime(0, audio.currentTime, 0.08);
        setTimeout(() => { if (!on) void audio.suspend(); }, 400);
      }
    },
    dispose() {
      disposed = true; on = false;
      document.removeEventListener("visibilitychange", onVisibility);
      void ctx?.close();
      ctx = null;
    },
  };
}
