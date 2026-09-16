import { defineConfig } from 'vite';

export default defineConfig({
  root: 'web',
  // `?raw` imports reach outside web/ into shared/spec/, so the dev server needs
  // permission to serve from the repo root.
  server: {
    fs: { allow: ['..'] },
    // `python3 serve.py` runs the PyTorch model here, so the client can compare runtimes
    // during development without a build step.
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: 'web/index.html',
        article: 'web/article.html',
        edit: 'web/edit.html',
      },
    },
  },
  test: {
    root: '.',
    include: ['ts/**/*.test.ts', 'web/**/*.test.ts'],
  },
});
