// Post-build: generate llms.txt / llms-full.txt / sitemap.xml and the legacy URL
// redirects, then sanity-check the dist output.
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const distDir = path.join(root, '.vitepress', 'dist');
const siteUrl = 'https://kuanghl.github.io/ai-data-center-systems';
const basePath = '/ai-data-center-systems';

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

const errors = [];
if (!existsSync(distDir)) errors.push(`dist directory missing: ${distDir}`);

// --- collect markdown pages -------------------------------------------------

const pages = []; // { file, rel, title, url }
function pageUrl(relDir, name) {
  // index.md -> directory URL, foo.md -> /foo/ (cleanUrls)
  const suffix = name === 'index.md' ? '' : `${name.replace(/\.md$/, '')}/`;
  return `${siteUrl}/${relDir ? `${relDir}/` : ''}${suffix}`;
}
function firstHeading(file, fallback) {
  const match = readFileSync(file, 'utf8').match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : fallback;
}
function collect(dir, relDir) {
  for (const entry of readdirSync(dir, { withFileTypes: true }).sort((a, b) =>
    a.name.localeCompare(b.name),
  )) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      collect(abs, relDir ? `${relDir}/${entry.name}` : entry.name);
      continue;
    }
    if (!entry.name.endsWith('.md')) continue;
    const rel = relDir ? `${relDir}/${entry.name}` : entry.name;
    pages.push({
      file: abs,
      rel,
      title: firstHeading(abs, entry.name),
      url: pageUrl(relDir, entry.name),
    });
  }
}
pages.push({
  file: path.join(root, 'index.md'),
  rel: 'index.md',
  title: 'AI Data Center Systems',
  url: `${siteUrl}/`,
});
for (const contentRoot of contentRoots) collect(path.join(root, contentRoot), contentRoot);

// --- llms.txt / llms-full.txt -------------------------------------------------

const description =
  'AI data center networking, LLM inference, training, MLOps, storage, and systems performance engineering study notes.';

let llms = `# AI Data Center Systems\n\n${description}\n\n## Docs\n\n`;
for (const page of pages) llms += `- [${page.title}](${page.url}) - ${page.rel}\n`;
writeFileSync(path.join(distDir, 'llms.txt'), llms);

let llmsFull = `# AI Data Center Systems\n\nThis file concatenates the source Markdown used to build the public documentation site.\n`;
for (const page of pages) {
  llmsFull += `\n## ${page.title}\n\nSource: ${page.rel}\nURL: ${page.url}\n\n${readFileSync(page.file, 'utf8').trim()}\n`;
}
writeFileSync(path.join(distDir, 'llms-full.txt'), llmsFull);

// --- sitemap.xml ---------------------------------------------------------------

const htmlFiles = [];
function collectHtml(dir, relDir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      collectHtml(abs, relDir ? `${relDir}/${entry.name}` : entry.name);
      continue;
    }
    if (entry.name.endsWith('.html') && entry.name !== '404.html') {
      htmlFiles.push({ abs, rel: relDir ? `${relDir}/${entry.name}` : entry.name });
    }
  }
}
collectHtml(distDir, '');

const sitemapUrls = htmlFiles.map(({ rel }) => {
  const dir = path.posix.dirname(rel);
  const name = path.posix.basename(rel);
  const route =
    name === 'index.html'
      ? dir === '.'
        ? '/'
        : `/${dir}/`
      : `/${dir === '.' ? '' : `${dir}/`}${name.replace(/\.html$/, '')}/`;
  return `${siteUrl}${basePath}${route}`;
});
const sitemap = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${sitemapUrls
  .map((url) => `  <url><loc>${url}</loc></url>`)
  .join('\n')}\n</urlset>\n`;
writeFileSync(path.join(distDir, 'sitemap.xml'), sitemap);

// --- legacy redirects -----------------------------------------------------------
// The site moved from a custom domain (flat routes) to this GitHub Pages project
// path. Old URLs keep working via meta-refresh pages generated here.

const legacyMappings = [
  { from: '/ai-data-center-network', to: '/network' },
  { from: '/efficient-llm-inference-systems', to: '/inference/efficient-llm-inference-systems' },
  { from: '/inference/appendix', to: '/inference/efficient-llm-inference-systems/appendix' },
  { from: '/inference/week01', to: '/inference/efficient-llm-inference-systems/week01' },
  { from: '/inference/week02', to: '/inference/efficient-llm-inference-systems/week02' },
  { from: '/inference/week03', to: '/inference/efficient-llm-inference-systems/week03' },
  { from: '/inference/week04', to: '/inference/efficient-llm-inference-systems/week04' },
  { from: '/inference/week05', to: '/inference/efficient-llm-inference-systems/week05' },
  { from: '/ai-system-performance-engineering', to: '/systems-performance' },
  { from: '/cme295', to: '/courses/cme295' },
  { from: '/deep-learning-for-network-engineers', to: '/courses/deep-learning-for-network-engineers' },
];

function redirectHtml(target) {
  return `<!doctype html>
<title>Redirecting to: ${target}</title>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0;url=${target}">
<meta name="robots" content="noindex">
<body><a href="${target}">Redirecting to <code>${target}</code></a></body>
`;
}

let redirectCount = 0;
for (const { from, to } of legacyMappings) {
  const newDir = path.join(distDir, to.slice(1));
  if (!existsSync(newDir)) {
    errors.push(`legacy mapping ${from} -> ${to}: target directory missing in dist`);
    continue;
  }
  for (const { abs } of htmlFiles) {
    if (!abs.startsWith(newDir + path.sep)) continue;
    const relDir = path.relative(newDir, path.dirname(abs));
    const relDirPosix = relDir === '.' ? '' : relDir.split(path.sep).join('/');
    const stem =
      path.basename(abs) === 'index.html' ? '' : `${path.basename(abs).replace(/\.html$/, '')}/`;
    const target = `${basePath}${to}/${relDirPosix ? `${relDirPosix}/` : ''}${stem}`;
    const outDir = path.join(distDir, from.slice(1), relDirPosix, stem);
    mkdirSync(outDir, { recursive: true });
    writeFileSync(path.join(outDir, 'index.html'), redirectHtml(target));
    redirectCount += 1;
  }
}

// --- sanity checks ----------------------------------------------------------------

for (const contentRoot of contentRoots) {
  if (!existsSync(path.join(distDir, contentRoot, 'index.html'))) {
    errors.push(`missing section page dist/${contentRoot}/index.html`);
  }
}
for (const file of [
  'llms.txt',
  'llms-full.txt',
  'sitemap.xml',
  'talks/sr-iov-with-dgx-b200/index.html',
]) {
  if (!existsSync(path.join(distDir, file))) errors.push(`missing dist file: ${file}`);
}

if (errors.length > 0) {
  console.error(`Post-build validation failed with ${errors.length} error(s):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exitCode = 1;
} else {
  console.log(
    `Post-build done: ${pages.length} pages indexed in llms.txt, ${sitemapUrls.length} sitemap URLs, ${redirectCount} legacy redirects.`,
  );
}
