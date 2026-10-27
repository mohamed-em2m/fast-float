# React + Vite

## Quick start

The frontend uses **pnpm** (pinned through Corepack in `package.json`), so the
lockfile and the installed tree are reproducible without a global install:

```bash
corepack enable pnpm        # one-time; setup.sh does this automatically
pnpm install --frozen-lockfile
pnpm dev                    # http://localhost:3000, proxies /api -> :8000
```

`pnpm install` reuses pnpm's content-addressable store, so reruns and extra
worktrees hardlink existing packages instead of downloading them again.

```bash
pnpm test        # Vitest + Testing Library
pnpm lint        # oxlint
pnpm build       # production build in dist/
pnpm preview     # serve the build through the same /api proxy
```

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.
