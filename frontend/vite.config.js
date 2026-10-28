import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Same-origin API proxy shared by `vite dev` and `vite preview`.
 *
 * The Studio calls `/api/*` and lets the dev/preview server forward it to the
 * FastAPI backend. The target used to be hard-coded to :8000, so
 * `PORT=9000 ./scripts/run_all.sh` (or a `npm run preview` build) produced a
 * Studio that reported "Disconnected" no matter what.
 *
 * Resolution order:
 *   1. VITE_API_PROXY_TARGET (full URL, e.g. http://127.0.0.1:9000)
 *   2. PORT (backend port, matching scripts/run_all.sh)
 *   3. http://localhost:8000
 */
function proxyTarget(env) {
  const explicit = process.env.VITE_API_PROXY_TARGET || env.VITE_API_PROXY_TARGET
  if (explicit?.trim()) return explicit.trim().replace(/\/$/, '')
  const port = process.env.PORT || env.PORT || '8000'
  return `http://localhost:${port}`
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = proxyTarget(env)
  const frontendPort = Number(process.env.FRONTEND_PORT || env.FRONTEND_PORT) || 3000

  const proxy = {
    '/api': {
      target,
      changeOrigin: true,
      ws: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  }

  return {
    plugins: [react()],
    build: {
      // The LiveKit SDK is a dedicated, lazily-loaded chunk (~535 kB) that only
      // users of the live transports fetch; the initial Studio chunk is ~320 kB.
      // The warning is about the initial payload, so set the limit just above
      // the SDK chunk instead of splitting a vendor library arbitrarily.
      chunkSizeWarningLimit: 600,
    },
    server: {
      port: frontendPort,
      allowedHosts: true,
      proxy,
    },
    // `pnpm preview` serves the production build; without a proxy there,
    // the built Studio could not reach the API at all.
    preview: {
      port: frontendPort,
      allowedHosts: true,
      proxy,
    },
  }
})
