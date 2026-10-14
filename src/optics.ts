import * as THREE from "three/webgpu";
import { Fn, exp, float, max, mix, normalize, refract, smoothstep, texture, uv, varyingProperty, vec2, vec3, vec4, wgslFn } from "three/tsl";
import { ripplesAt } from "./ripples";
import { SUN, WORLD_M } from "./light";
import { SURF, bedAt, incoming } from "./surf";

type F = THREE.Node<"float">;
type V2 = THREE.Node<"vec2">;
type V4 = THREE.Node<"vec4">;

/** Water's refractive index for green light; the material spreads red and blue either side of it (see DISPERSION). */
const IOR = 1.333;
/** How far red and blue land apart per metre of water, over the screen height in metres: true dispersion (1.331 to
 * 1.338) widened by about two thirds, so a caustic's sharpest folds show a faint coloured fringe. Drawn as one pass
 * and sampled three times, a third of the cost of three passes with the same look. */
export const DISPERSION = 0.0053;

const lightDensity = wgslFn(/* wgsl */ `
fn swashLightDensity(source: vec2f, receiver: vec2f) -> f32 {
  let sx = dpdx(source);
  let sy = dpdy(source);
  let rx = dpdx(receiver);
  let ry = dpdy(receiver);
  let sourceArea = abs(sx.x * sy.y - sx.y * sy.x);
  let receiverArea = abs(rx.x * ry.y - rx.y * ry.x);
  // A flat source/receiver pair has density 1. The bound keeps a subpixel fold from becoming an infinite
  // needle; additive overlap still sums separate light paths.
  return clamp(sourceArea / max(receiverArea, 1e-10), 0.0, 8.0);
}
`);

type OpticalUniforms = {
  time: THREE.UniformNode<"float", number>;
  aspect: THREE.UniformNode<"float", number>;
  cam: THREE.UniformNode<"float", number>;
};

/** Screen-height world units to metres on the simulated strip (its middle under the middle of the view), and to the
 * strip's texture coordinates. */
export const toMetres = (w: V2, aspect: F) => vec2(w.x.sub(aspect.mul(0.5)).mul(WORLD_M).add(SURF.W / 2), w.y.mul(WORLD_M));
export const stripUV = (m: V2) => vec2(m.x.div(SURF.W), m.y.sub(SURF.Y0).div(SURF.Y1 - SURF.Y0));
/** How much of the view at metres y is the incoming sea rather than the simulated strip (its relaxation zone). */
export const offStrip = (y: F) => smoothstep(SURF.Y1 - 0.6, SURF.Y1 - 0.15, y);

/** The water's surface at metres m: height above still water, slope (both from the solver, or the incoming sea
 * offshore) and depth over the beach. */
export function waterAt(surface: THREE.Texture, m: V2, t: F): { eta: F; slope: V2; depth: F } {
  const s = texture(surface, stripUV(m)).level(float(0)) as unknown as V4;
  const off = offStrip(m.y);
  const sea = incoming(m.x, m.y, t);
  const eta = mix(s.x, sea.eta, off);
  return { eta, slope: mix(s.yz, sea.slope, off), depth: eta.sub(bedAt(m.x, m.y)) };
}

/** Where sunlight entering the surface at `source` lands on the bed, world units. */
function project(surface: THREE.Texture, ripples: THREE.Texture, wake: THREE.Texture, source: V2, aspect: F, t: F): V2 {
  const m = toMetres(source, aspect);
  const water = waterAt(surface, m, t);
  const depth = max(water.depth, 0).toVar();
  const r = ripplesAt(ripples, source);
  const wind = r.xy.add(r.zw.mul(exp(depth.div(-0.12)))).mul(smoothstep(0.002, 0.025, depth));
  // A bore's face spans a cell or two on the grid; under its roller it scatters rather than focuses the light.
  const hand = (texture(wake, stripUV(m)).level(float(0)) as unknown as V4).yz.mul(offStrip(m.y).oneMinus());
  const slope = water.slope.div(water.slope.length().mul(2.2).add(1)).add(wind).add(hand).toVar();
  const ray = refract(vec3(-SUN.x, -SUN.y, -SUN.z), normalize(vec3(slope.negate(), 1)), 1 / IOR).toVar();
  return source.add(ray.xy.mul(depth.div(WORLD_M)).div(max(ray.z.negate(), 0.2)));
}

/** The source point whose flat-water ray lands on `receiver` (two fixed-point steps over still water). */
function sourceFor(receiver: V2, aspect: F): V2 {
  const flat = new THREE.Vector3(-SUN.x, -SUN.y, -SUN.z);
  const eta = 1 / IOR, c = -flat.z, k = 1 - eta * eta * (1 - c * c);
  flat.multiplyScalar(eta).add(new THREE.Vector3(0, 0, eta * c - Math.sqrt(k)));
  const travel = vec2(flat.x, flat.y).div(Math.max(-flat.z, 0.2) * WORLD_M);
  const source = receiver.toVar();
  for (let i = 0; i < 2; i++) {
    const m = toMetres(source, aspect);
    source.assign(receiver.sub(travel.mul(max(bedAt(m.x, m.y).negate(), 0))));
  }
  return source;
}

/**
 * Receiver-space caustics: project a triangulated sunlight wavefront through the wave surface onto the bed and
 * measure how its area changes. Evan Wallace's projected-mesh method, written here in TSL:
 * https://medium.com/@evanwallace/rendering-realtime-caustics-in-webgl-2a99a29a0b2c
 * (reference implementation, MIT: https://github.com/evanw/webgl-water/blob/master/renderer.js).
 * One pass into a half-float texture about the size of the view. The surface is the simulated water, so a bore's
 * steep front and the swash's edge focus the light into bright lines on the sand, as they do on a real beach.
 */
export function createCaustics(renderer: THREE.WebGPURenderer, u: OpticalUniforms, surface: THREE.Texture, ripples: THREE.Texture, wake: THREE.Texture): {
  texture: THREE.Texture;
  resize: (width: number, height: number) => void;
  update: () => void;
  dispose: () => void;
} {
  const target = new THREE.RenderTarget(512, 512, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
    colorSpace: THREE.NoColorSpace,
  });
  target.texture.name = "Swash receiver-space caustic irradiance";
  // Vertices about 5 mm of beach apart (0.0055 screen heights), whatever the view's shape.
  const grid = (aspect: number) => new THREE.PlaneGeometry(1, 1, Math.min(420, Math.max(64, Math.round(1.3 * aspect / 0.0055))), 236);
  let geometry = grid(1), gridAspect = 1;
  const scene = new THREE.Scene();
  const material = new THREE.MeshBasicNodeMaterial({
    depthTest: false, depthWrite: false, side: THREE.DoubleSide, transparent: true,
    blending: THREE.AdditiveBlending, toneMapped: false, forceSinglePass: true,
  });
  const sourceWorld = varyingProperty("vec2", "swashLightSource");
  const receiverWorld = varyingProperty("vec2", "swashLightReceiver");
  material.vertexNode = Fn(() => {
    // PlaneGeometry UV is bottom-up; world and receiver UV here are top-down.
    const q = vec2(uv().x, float(1).sub(uv().y)).mul(1.3).sub(0.15);
    const nominal = vec2(q.x.mul(u.aspect), q.y.add(u.cam));
    sourceWorld.assign(sourceFor(nominal, u.aspect as unknown as F));
    receiverWorld.assign(project(surface, ripples, wake, sourceWorld, u.aspect as unknown as F, u.time as unknown as F));
    return vec4(receiverWorld.x.div(u.aspect).mul(2).sub(1), float(1).sub(receiverWorld.y.sub(u.cam).mul(2)), 0, 1);
  })();
  const density = lightDensity(sourceWorld, receiverWorld) as unknown as F;
  material.fragmentNode = vec4(density, density, density, 1);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  scene.add(mesh);
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
  camera.position.z = 1;
  const clearColour = new THREE.Color();
  let disposed = false;

  return {
    texture: target.texture,
    resize(width, height) {
      // About one texel per CSS pixel, at most 1024 on the long side.
      const k = Math.min(1, 1024 / Math.max(width, height));
      target.setSize(Math.max(64, Math.round(width * k)), Math.max(64, Math.round(height * k)));
      const aspect = width / height;
      if (Math.abs(aspect - gridAspect) > 0.08) {
        geometry.dispose(); geometry = grid(aspect); gridAspect = aspect;
        mesh.geometry = geometry;
      }
    },
    update() {
      if (disposed) return;
      const previousTarget = renderer.getRenderTarget();
      renderer.getClearColor(clearColour);
      const previousAlpha = renderer.getClearAlpha();
      const previousAutoClear = renderer.autoClear;
      try {
        renderer.autoClear = false;
        renderer.setRenderTarget(target);
        renderer.setClearColor(0x000000, 0);
        renderer.clear(true, false, false);
        renderer.render(scene, camera);
      } finally {
        renderer.setRenderTarget(previousTarget);
        renderer.setClearColor(clearColour, previousAlpha);
        renderer.autoClear = previousAutoClear;
      }
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      scene.clear();
      geometry.dispose();
      material.dispose();
      target.dispose();
    },
  };
}
