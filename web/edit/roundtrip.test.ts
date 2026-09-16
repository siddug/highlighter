/**
 * @vitest-environment happy-dom
 *
 * Loading the article into TipTap and saving it back must not lose anything.
 *
 * This is the test that matters most here. ProseMirror silently discards any element its
 * schema does not recognise, so without RawBlock the first save would delete every SVG
 * figure, callout and definition box — 141 KB of article replaced by its paragraphs, with
 * no error anywhere. These assertions are what make the editor safe to point at real work.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Editor } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import { TableKit } from '@tiptap/extension-table';
import { beforeAll, describe, expect, it } from 'vitest';
import { KeepAttributes, PreservedSpan, RawBlock, serializeWithRaw } from './rawblock';

const CONTENT = resolve(dirname(fileURLToPath(import.meta.url)), '../article.content.html');
const original = readFileSync(CONTENT, 'utf8');

function makeEditor(content: string): Editor {
  return new Editor({
    element: document.createElement('div'),
    extensions: [
      StarterKit.configure({ heading: { levels: [2, 3, 4] } }),
      TableKit.configure({ table: { resizable: false } }),
      RawBlock,
      KeepAttributes,
      PreservedSpan,
    ],
    content,
  });
}

let editor: Editor;
let saved: string;

beforeAll(() => {
  editor = makeEditor(original);
  saved = serializeWithRaw(editor);
});

const count = (html: string, needle: string) => html.split(needle).length - 1;

describe('article round-trip through the editor', () => {
  it('keeps every SVG figure', () => {
    expect(count(saved, '<svg')).toBe(count(original, '<svg'));
    expect(count(saved, '</svg>')).toBe(count(original, '</svg>'));
  });

  it('keeps every callout, definition, worked example and budget strip', () => {
    for (const cls of ['callout', 'define', 'worked', 'budget']) {
      expect(count(saved, `class="${cls}"`), cls).toBe(count(original, `class="${cls}"`));
    }
  });

  it('keeps the table of contents', () => {
    expect(count(saved, 'class="toc"')).toBe(1);
  });

  it('keeps every table', () => {
    expect(count(saved, '<table')).toBe(count(original, '<table'));
    expect(count(saved, '<tr')).toBe(count(original, '<tr'));
  });

  it('keeps every section heading', () => {
    expect(count(saved, '<h2')).toBe(count(original, '<h2'));
    expect(count(saved, '<h3')).toBe(count(original, '<h3'));
  });

  it('keeps every code block', () => {
    expect(count(saved, '<pre')).toBe(count(original, '<pre'));
  });

  it('retains essentially all of the prose', () => {
    const words = (html: string) =>
      html.replace(/<[^>]+>/g, ' ').split(/\s+/).filter((w) => w.length > 2).length;
    // Serialization normalizes whitespace and attribute order, so byte equality is not
    // available; losing content is what we actually need to rule out.
    expect(words(saved) / words(original)).toBeGreaterThan(0.98);
  });

  it('survives a second pass unchanged', () => {
    // Idempotence: if load/save were lossy, a second cycle would compound the loss.
    const twice = serializeWithRaw(makeEditor(saved));
    expect(count(twice, '<svg')).toBe(count(saved, '<svg'));
    expect(count(twice, '<table')).toBe(count(saved, '<table'));
    expect(twice.length / saved.length).toBeGreaterThan(0.99);
  });

  it('keeps the styling classes on ordinary blocks', () => {
    // Counting tags alone missed all of this the first time: `p.filename`, the part
    // dividers and the reading demos all round-tripped as bare paragraphs.
    for (const cls of ['filename', 'part-head', 'reading-demo', 'telegram', 'dim']) {
      expect(count(saved, `"${cls}"`), cls).toBe(count(original, `"${cls}"`));
    }
  });

  it('keeps inline spans inside headings', () => {
    // <span class="num">07</span> inside each h2. Losing it renders "07The local window".
    expect(count(saved, 'class="num"')).toBe(count(original, 'class="num"'));
  });

  it('keeps syntax colouring inside code blocks', () => {
    // ProseMirror code blocks hold plain text, so these can only survive as raw blocks.
    for (const cls of ['class="k"', 'class="c"', 'class="s"']) {
      expect(count(saved, cls), cls).toBe(count(original, cls));
    }
  });

  it('keeps the heading anchors the table of contents links to', () => {
    for (const id of ['id="s1"', 'id="s8"', 'id="s18"']) {
      expect(count(saved, id), id).toBe(1);
    }
  });

  it('preserves specific hard-won numbers', () => {
    for (const fact of ['39,361', '0.936', '0.704', '23,584', '78×']) {
      expect(saved, fact).toContain(fact);
    }
  });
});
