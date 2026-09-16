/**
 * TipTap editor over the article, persisting to disk through the server.
 *
 * Saving to localStorage would make this a private notepad. Writing through to
 * web/article.content.html is what makes it co-writing: whatever you type here is a file
 * on disk that the other author can read, diff and build on.
 */

import { Editor } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import { TableKit } from '@tiptap/extension-table';
import { KeepAttributes, PreservedSpan, RawBlock, serializeWithRaw } from './rawblock';

const statusEl = document.getElementById('status')!;
const saveBtn = document.getElementById('save') as HTMLButtonElement;

type State = 'clean' | 'dirty' | 'saving' | 'error';

function setStatus(state: State, detail = ''): void {
  const labels: Record<State, string> = {
    clean: 'saved',
    dirty: 'unsaved changes',
    saving: 'saving…',
    error: 'save failed',
  };
  statusEl.textContent = detail ? `${labels[state]} — ${detail}` : labels[state];
  statusEl.dataset.state = state;
  saveBtn.disabled = state === 'saving';
}

async function loadContent(): Promise<string> {
  const response = await fetch('/api/article');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()).content;
}

let saving = false;
let pending = false;

async function save(editor: Editor): Promise<void> {
  if (saving) {
    pending = true;
    return;
  }
  saving = true;
  setStatus('saving');
  try {
    const response = await fetch('/api/article', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: serializeWithRaw(editor) }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(body.error ?? `HTTP ${response.status}`);
    }
    const { bytes } = await response.json();
    setStatus('clean', `${(bytes / 1024).toFixed(0)} KB`);
  } catch (error) {
    setStatus('error', (error as Error).message);
  } finally {
    saving = false;
    if (pending) {
      pending = false;
      void save(editor);
    }
  }
}

async function boot(): Promise<void> {
  let content: string;
  try {
    content = await loadContent();
  } catch (error) {
    setStatus('error', `could not load — is serve.py running? (${(error as Error).message})`);
    return;
  }

  let debounce: number | undefined;

  const editor = new Editor({
    element: document.getElementById('editor')!,
    extensions: [
      // The article has its own heading rhythm and code blocks; keep those, drop the
      // editor's opinionated extras.
      StarterKit.configure({ heading: { levels: [2, 3, 4] } }),
      TableKit.configure({ table: { resizable: false } }),
      RawBlock,
      KeepAttributes,
      PreservedSpan,
    ],
    content,
    onCreate: () => setStatus('clean'),
    onUpdate: () => {
      setStatus('dirty');
      window.clearTimeout(debounce);
      debounce = window.setTimeout(() => void save(editor), 1200);
    },
  });

  saveBtn.addEventListener('click', () => void save(editor));

  // Ctrl/Cmd-S is muscle memory; intercept it rather than letting the browser save the page.
  addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 's') {
      event.preventDefault();
      void save(editor);
    }
  });

  addEventListener('beforeunload', (event) => {
    if (statusEl.dataset.state === 'dirty' || statusEl.dataset.state === 'saving') {
      event.preventDefault();
    }
  });
}

void boot();
