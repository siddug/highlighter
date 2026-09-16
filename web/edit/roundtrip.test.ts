/**
 * @vitest-environment happy-dom
 *
 * Loading the article into TipTap and saving it back must not lose anything.
 *
 * The schema is deliberately small — paragraphs, headings, lists, blockquotes, tables,
 * code blocks, images — because the article's destination is a blog that understands only
 * those. Anything the schema does not recognise is silently dropped by ProseMirror, with
 * no error, so these assertions are what make the editor safe to point at real work.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Editor } from '@tiptap/core';
import { beforeAll, describe, expect, it } from 'vitest';
import { extensions } from './schema';

const CONTENT = resolve(dirname(fileURLToPath(import.meta.url)), '../article.content.html');
const original = readFileSync(CONTENT, 'utf8');

const makeEditor = (content: string) =>
  new Editor({ element: document.createElement('div'), extensions, content });

let saved: string;
beforeAll(() => {
  saved = makeEditor(original).getHTML();
});

const count = (html: string, needle: string) => html.split(needle).length - 1;

describe('article round-trip through the editor', () => {
  it('uses only portable elements', () => {
    // The point of the conversion: nothing here should need a custom node to survive.
    const tags = new Set([...saved.matchAll(/<([a-z][a-z0-9]*)/g)].map((m) => m[1]!));
    const portable = new Set([
      'p', 'h2', 'h3', 'h4', 'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
      'table', 'thead', 'tbody', 'tr', 'th', 'td', 'colgroup', 'col',
      'img', 'a', 'strong', 'em', 'mark', 'br', 's', 'u',
    ]);
    expect([...tags].filter((t) => !portable.has(t))).toEqual([]);
  });

  it('keeps every image', () => {
    expect(count(saved, '<img')).toBe(count(original, '<img'));
    expect(count(saved, './figures/')).toBe(count(original, './figures/'));
  });

  it('keeps every blockquote, which is where the callouts went', () => {
    expect(count(saved, '<blockquote')).toBe(count(original, '<blockquote'));
  });

  it('keeps every table and row', () => {
    expect(count(saved, '<table')).toBe(count(original, '<table'));
    expect(count(saved, '<tr')).toBe(count(original, '<tr'));
  });

  it('keeps every heading and code block', () => {
    expect(count(saved, '<h2')).toBe(count(original, '<h2'));
    expect(count(saved, '<h3')).toBe(count(original, '<h3'));
    expect(count(saved, '<pre')).toBe(count(original, '<pre'));
  });

  it('keeps heading anchors', () => {
    // Without an explicit extension these vanish on the first save, taking every deep
    // link with them — and silently, since ProseMirror drops unknown attributes.
    expect(count(saved, 'id="')).toBe(count(original, 'id="'));
  });

  it('keeps the highlighted words in the opening figure', () => {
    expect(count(saved, '<mark')).toBe(count(original, '<mark'));
  });

  it('retains essentially all of the prose', () => {
    const words = (html: string) =>
      html.replace(/<[^>]+>/g, ' ').split(/\s+/).filter((w) => w.length > 2).length;
    expect(words(saved) / words(original)).toBeGreaterThan(0.98);
  });

  it('survives a second pass unchanged', () => {
    const twice = makeEditor(saved).getHTML();
    expect(twice).toBe(saved);
  });

  it('preserves specific hard-won numbers', () => {
    for (const fact of ['39,361', '0.936', '0.704', '23,584', '78×']) {
      expect(saved, fact).toContain(fact);
    }
  });
});
