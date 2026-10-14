// Builds the demo into docs/ (GitHub Pages): the bundle, its CSS and font, the page, the sand maps, the poster and
// the recording shown where there is no WebGPU.
import { build } from "esbuild";
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";

rmSync("docs", { recursive: true, force: true });
mkdirSync("docs", { recursive: true });
await build({
  entryPoints: ["demo/main.ts"], bundle: true, minify: true, format: "esm", target: "es2022", outdir: "docs",
  loader: { ".woff2": "file", ".woff": "file" }, assetNames: "fonts/[name]-[hash]", legalComments: "none",
});
cpSync("demo/index.html", "docs/index.html");
cpSync("demo/sand", "docs/sand", { recursive: true });
for (const file of ["poster.webp", "poster-tall.webp", "swash.mp4", "swash-tall.mp4", "swash.webm", "swash-tall.webm"]) if (existsSync(`demo/${file}`)) cpSync(`demo/${file}`, `docs/${file}`);
