// Builds the public/ directory that VitePress serves verbatim:
//   - static/            hand-maintained files (robots.txt, ads.txt, favicon, translate-init.js)
//   - talks/             self-contained slide-deck bundles (HTML/CSS/JS/fonts)
//   - fabric.svg         referenced as /fabric.svg by the home page
//   - <contentRoot>/...  linked assets (images, PDFs) copied next to their pages
import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const publicDir = path.join(root, 'public');

const assetExtensions = new Set(['.gif', '.jpeg', '.jpg', '.pdf', '.png', '.svg', '.webp']);
const contentRoots = [
  'network',
  'gpu',
  'training',
  'inference',
  'mlops',
  'storage',
  'systems-performance',
  'courses',
];

rmSync(publicDir, { recursive: true, force: true });
mkdirSync(publicDir, { recursive: true });

cpSync(path.join(root, 'static'), publicDir, { recursive: true });
cpSync(path.join(root, 'talks'), path.join(publicDir, 'talks'), { recursive: true });
cpSync(path.join(root, 'fabric.svg'), path.join(publicDir, 'fabric.svg'));

// Extensionless files linked directly from the markdown (e.g. `dmon` logs,
// LICENSE) are published next to their pages so the links keep working.
const linkedNoExt = new Set();
function scanFile(file) {
  const dir = path.dirname(file);
  const text = readFileSync(file, 'utf8');
  for (const match of text.matchAll(/\]\(([^)\s]+)\)/g)) {
    const ref = match[1];
    if (/^(https?:|\/\/|\/|#|data:)/.test(ref)) continue;
    const clean = ref.split('#')[0].replace(/^\.\//, '');
    if (!clean || path.extname(clean)) continue;
    linkedNoExt.add(path.relative(root, path.join(dir, clean)));
  }
}
function scanDir(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) scanDir(abs);
    else if (entry.name.endsWith('.md')) scanFile(abs);
  }
}
for (const contentRoot of contentRoots) scanDir(path.join(root, contentRoot));
scanFile(path.join(root, 'index.md'));

function copyAssets(dir, outDir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const from = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      copyAssets(from, path.join(outDir, entry.name));
      continue;
    }
    const isAsset = assetExtensions.has(path.extname(entry.name).toLowerCase());
    const isLinkedNoExt = linkedNoExt.has(path.relative(root, from));
    if (!isAsset && !isLinkedNoExt) continue;
    const to = path.join(outDir, entry.name);
    mkdirSync(path.dirname(to), { recursive: true });
    cpSync(from, to);
  }
}

for (const contentRoot of contentRoots) {
  copyAssets(path.join(root, contentRoot), path.join(publicDir, contentRoot));
}

console.log(`public/ ready: static files, talks, and linked assets for ${contentRoots.length} content roots.`);
