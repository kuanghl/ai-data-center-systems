# AI Data Center Systems

AI data center networking, LLM inference, training, MLOps, storage, and systems
performance engineering study notes, published as a [VitePress](https://vitepress.dev)
documentation site.

Live site: <https://kuanghl.github.io/ai-data-center-systems>

## Content structure

| Section | Path | Contents |
|---|---|---|
| Network | `network/` | `chap01`–`chap12` + appendices (RDMA examples, Clos/EVPN-BGP lab, failure analysis) |
| GPU & Accelerators | `gpu/` | Courses (PMPP, CS149, GPU MODE, Heterogeneous Systems) + appendix + labs |
| Training | `training/` | MLPerf Training workloads |
| Inference | `inference/` | Efficient LLM Inference Systems (week01–week05), FlashAttention, SGLang, model profiles |
| MLOps | `mlops/` | ML release, lineage, CI/CD/CT, deployment, monitoring |
| Storage | `storage/` | ZFS tuning, MLPerf Storage |
| Systems Performance | `systems-performance/` | `chap01`–`chap05` + labs |
| Courses | `courses/` | CME295 (lec-01–lec-06), Deep Learning for Network Engineers (week01–week04) |
| Talks | `talks/` | Self-contained slide decks (served verbatim, not rendered by VitePress) |

Each section root `index.md` is the chapter table of contents. The sidebar is
generated automatically from the directory tree in `.vitepress/config.mts`,
ordered chapter-first (`chapNN` / `weekNN` / `lecNN` in numeric order, then
appendices and articles).

## Local development

Requires Node.js 20+ (CI uses Node 24).

```sh
npm ci          # install dependencies
npm run dev     # dev server with hot reload
```

Open <http://localhost:5173/ai-data-center-systems/> (the site uses the
`/ai-data-center-systems/` base path, same as the production URL).

## Local build

```sh
npm run build
```

This runs three steps:

1. `prebuild` (`scripts/prebuild.mjs`) — regenerates `public/`: static files,
   the `talks/` slide bundles, and every image/PDF linked from the Markdown
   (`public/` is gitignored and is never edited by hand).
2. `vitepress build` — renders the site into `.vitepress/dist/` and fails the
   build on dead links.
3. `postbuild` (`scripts/postbuild.mjs`) — writes `llms.txt`, `llms-full.txt`,
   `sitemap.xml`, and legacy-URL redirect pages into `dist/`, then sanity-checks
   the output.

Preview the production build locally:

```sh
npm run build
npx vitepress preview
```

## Deployment (GitHub Pages)

Deployment uses the standard VitePress Pages workflow
(`.github/workflows/deploy-pages.yml`): on every push to `master` (or manual
dispatch) the workflow runs `npm ci && npm run build` and publishes
`.vitepress/dist/` through `actions/upload-pages-artifact` +
`actions/deploy-pages`.

One-time repo setup: **Settings → Pages → Source → GitHub Actions**.
After that, each successful push deploys automatically; the site is served at
`https://<username>.github.io/ai-data-center-systems/`.
