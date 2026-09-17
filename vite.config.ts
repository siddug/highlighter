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
      // The editor saves through to disk via `serve.py`, and the article is published on
      // siddg.com. Neither belongs on the public deploy: the write endpoint does not
      // exist there, so every save would fail silently.
      input: process.env.DEPLOY
        ? { main: 'web/index.html' }
        : {
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
