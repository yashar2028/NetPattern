import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// File-change events don't reliably reach containers through Windows bind mounts,
// so docker compose sets VITE_USE_POLLING=true.
const usePolling = process.env.VITE_USE_POLLING === 'true'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    watch: usePolling ? { usePolling: true, interval: 300 } : undefined,
  },
})
