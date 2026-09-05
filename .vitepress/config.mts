import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { withMermaid } from 'vitepress-plugin-mermaid';
import markdownItKatex from 'markdown-it-katex';
import { defineConfig } from 'vitepress';

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const siteUrl = 'https://kuanghl.github.io/ai-data-center-systems';
const basePath = '/ai-data-center-systems';
const googleAdSenseClient = process.env.PUBLIC_GOOGLE_ADSENSE_CLIENT ?? 'ca-pub-8128231647578658';
const googleAnalyticsId = process.env.PUBLIC_GA_MEASUREMENT_ID;

const contentRoots = [
  { dir: 'network', label: 'Network' },
  { dir: 'gpu', label: 'GPU & Accelerators' },
  { dir: 'training', label: 'Training' },
  { dir: 'inference', label: 'Inference' },
  { dir: 'mlops', label: 'MLOps' },
  { dir: 'storage', label: 'Storage' },
  { dir: 'systems-performance', label: 'Systems Performance' },
  { dir: 'courses', label: 'Courses' },
];

function firstHeading(file: string, fallback: string): string {
  const match = readFileSync(file, 'utf8').match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : fallback;
}

// Sidebar is generated from the content tree: every directory with an index.md is a
// collapsible group, every other .md file is a page item.
//
// Ordering is chapter-first: entries whose name starts with a letter prefix and a
// number (chap01, week02, lec-03, ch04, ...) sort before everything else, in
// numeric order, so chapters appear in reading order and appendices/articles
// fall behind them.
function sequenceKey(name: string): [number, number] {
  const match = name.match(/^([A-Za-z]+)-?(\d+)/);
  return match ? [0, Number(match[2])] : [1, 0];
}
function chapterCompare(a: string, b: string): number {
  const [ak, an] = sequenceKey(a);
  const [bk, bn] = sequenceKey(b);
  if (ak !== bk) return ak - bk;
  if (an !== bn) return an - bn;
  return a.localeCompare(b);
}

function buildGroup(dir: string, base: string, label: string): any {
  const abs = path.join(siteRoot, dir);
  const items: any[] = [];
  const indexFile = path.join(abs, 'index.md');
  if (existsSync(indexFile)) items.push({ text: firstHeading(indexFile, label), link: base });
  const entries = readdirSync(abs, { withFileTypes: true }).sort((a, b) =>
    chapterCompare(a.name, b.name),
  );
  for (const entry of entries) {
    if (entry.isDirectory()) {
      const subBase = `${base}${entry.name}/`;
      const subIndex = path.join(abs, entry.name, 'index.md');
      const subLabel = existsSync(subIndex) ? firstHeading(subIndex, entry.name) : entry.name;
      const sub = buildGroup(`${dir}/${entry.name}`, subBase, subLabel);
      if (sub) items.push(sub);
    } else if (entry.name.endsWith('.md') && entry.name !== 'index.md') {
      const name = entry.name.replace(/\.md$/, '');
      const file = path.join(abs, entry.name);
      items.push({ text: firstHeading(file, name), link: `${base}${name}/` });
    }
  }
  return items.length > 0 ? { text: label, collapsed: true, items } : null;
}

const sidebar = [
  ...contentRoots.map(({ dir, label }) => buildGroup(dir, `/${dir}/`, label)).filter(Boolean),
  {
    text: 'Talks',
    items: [{ text: 'SR-IOV with DGX B200', link: '/talks/sr-iov-with-dgx-b200/' }],
  },
];

const head: any[] = [
  ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
  ['meta', { property: 'og:title', content: 'AI Data Center Systems' }],
  ['meta', { property: 'og:type', content: 'website' }],
  ['meta', { name: 'twitter:card', content: 'summary' }],
];
if (googleAnalyticsId) {
  head.push(['script', { async: true, src: `https://www.googletagmanager.com/gtag/js?id=${googleAnalyticsId}` }]);
  head.push([
    'script',
    {
      innerHTML: `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${googleAnalyticsId}');`,
    },
  ]);
}
head.push(['script', { async: true, crossorigin: 'anonymous', src: `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${googleAdSenseClient}` }]);
// translate.js (https://translate.zvo.cn): client-side EN <-> zh-CN switching
// via the built-in language dropdown (mounted into the nav bar by
// public/translate-init.js).
head.push(['script', { src: 'https://cdn.staticfile.net/translate.js/3.18.66/translate.js' }]);
head.push(['script', { src: `${basePath}/translate-init.js` }]);

export default withMermaid(
  defineConfig({
    base: `${basePath}/`,
    siteUrl,
    cleanUrls: true,
    vite: {
      // mermaid is imported on every page via the plugin's patched app entry.
      // Without an explicit include it is not pre-bundled in dev, and the raw
      // CJS `fastdom` import crashes the module graph (blank white page).
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
    lang: 'en',
    title: 'AI Data Center Systems',
    description:
      'AI data center networking, LLM inference, training, MLOps, storage, and systems performance engineering study notes.',
    head,
    lastUpdated: true,
    // Extensionless files (LICENSE, dmon logs) are served from public/ by
    // scripts/prebuild.mjs; VitePress cannot see public files in its link check.
    ignoreDeadLinks: [
      /jax-scaling-book\/LICENSE$/,
      /labs-01-sweep-512\/dmon$/,
    ],
    srcExclude: [
      'README.md',
      'AGENTS.md',
      'PRODUCT.md',
      'talks/**',
      'ai-data-center-systems-main/**',
      'artifact/**',
      'refs/**',
      '.impeccable/**',
      'static/**',
    ],
    themeConfig: {
      favicon: '/favicon.svg',
      // Keep the top nav minimal: section navigation lives in the sidebar
      // (mirrors the old Starlight site, whose header had no section links).
      // The default theme has no desktop nav overflow handling, so a long
      // nav row pushes the trailing nav-bar-content-after slot (the
      // translate.js language dropdown) out of the viewport.
      nav: [{ text: 'GitHub', link: 'https://github.com/kuanghl/ai-data-center-systems' }],
      sidebar,
      darkModeSwitchLabel: 'Theme',
      lightModeSwitchTitle: 'Switch to light theme',
      darkModeSwitchTitle: 'Switch to dark theme',
      search: { provider: 'local' },
      footer: {
        message: 'AI data center systems study notes',
        copyright: 'kuanghl',
      },
    },
    markdown: {
      config: (md) => {
        // markdown-it-katex parses $/$$ (handling escapes, delimiters, and
        // "$5 and $10" correctly) and renders KaTeX HTML at build time; its
        // CSS is imported in theme/index.ts.
        md.use(markdownItKatex as any, { throwOnError: false });
        // ` ```math ` fences hold LaTeX, not code. Reroute them to the same
        // math_block token the $/$$ path uses so they render as KaTeX display
        // math instead of hitting Shiki as an unknown "math" code language
        // (which logs "The language 'math' is not loaded" per block).
        const fenceRule = md.block.ruler
          .getRules('')
          .find((fn: any) => fn.name === 'fence');
        md.block.ruler.disable('fence');
        md.block.ruler.before('code', 'math-fence', (state: any, startLine: number, endLine: number, silent: boolean) => {
          if (!fenceRule(state, startLine, endLine, silent)) return false;
          const token = state.tokens[state.tokens.length - 1];
          if (token.info.trim() === 'math') {
            token.type = 'math_block';
            token.tag = '';
            token.nesting = 0;
            token.markup = '';
          }
          return true;
        });
      },
    },
    mermaid: {
      securityLevel: 'loose',
    },
  }),
);
