import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  preview: { port: 4173 },
  define: { 'import.meta.env.VITE_API_URL': JSON.stringify(process.env.VITE_API_URL || '') },
  server: {
    port: 5173,
    proxy: { '/api': 'http://localhost:8000' },
  },
})
