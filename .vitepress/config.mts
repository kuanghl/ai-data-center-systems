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
function buildGroup(dir: string, base: string, label: string): any {
  const abs = path.join(siteRoot, dir);
  const items: any[] = [];
  const indexFile = path.join(abs, 'index.md');
  if (existsSync(indexFile)) items.push({ text: firstHeading(indexFile, label), link: base });
  const entries = readdirSync(abs, { withFileTypes: true }).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
  for (const entry of entries) {
    if (entry.isDirectory()) {
      const subBase = `${base}${entry.name}/`;
      const subIndex = path.join(abs, entry.name, 'index.md');
      const subLabel = existsSync(subIndex) ? firstHeading(subIndex, entry.name) : entry.name;
      items.push(buildGroup(`${dir}/${entry.name}`, subBase, subLabel));
    } else if (entry.name.endsWith('.md') && entry.name !== 'index.md') {
      const name = entry.name.replace(/\.md$/, '');
      const file = path.join(abs, entry.name);
      items.push({ text: firstHeading(file, name), link: `${base}${name}/` });
    }
  }
  return { text: label, collapsed: true, items };
}

const sidebar = [
  ...contentRoots.map(({ dir, label }) => buildGroup(dir, `/${dir}/`, label)),
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
// translate.js (https://translate.zvo.cn): client-side EN -> zh-CN switching, driven by the navbar button.
head.push(['script', { src: 'https://cdn.staticfile.net/translate.js/3.18.66/translate.js' }]);
head.push(['script', { src: '/translate-init.js' }]);

export default withMermaid(
  defineConfig({
    base: `${basePath}/`,
    siteUrl,
    cleanUrls: true,
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
      'AGENTS.md',
      'PRODUCT.md',
      'talks/**',
      'ai-data-center-systems-main/**',
      'artifact/**',
      'refs/**',
      '.impeccable/**',
      'static/**',
    ],
    theme: {
      favicon: '/favicon.svg',
      nav: [
        ...contentRoots.map(({ dir, label }) => ({ text: label, link: `/${dir}/` })),
        { text: 'Talks', link: '/talks/sr-iov-with-dgx-b200/' },
        { text: 'GitHub', link: 'https://github.com/kuanghl/ai-data-center-systems' },
      ],
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
        md.use(markdownItKatex as any, { throwOnError: false });
      },
    },
    mermaid: {
      securityLevel: 'loose',
    },
  }),
);
