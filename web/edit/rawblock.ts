/**
 * A TipTap node that preserves arbitrary HTML verbatim.
 *
 * The article contains eleven hand-built SVG figures, callout boxes, definition boxes,
 * worked examples and parameter-budget strips. TipTap's schema does not know any of them,
 * and anything a ProseMirror schema does not recognise is silently discarded on load —
 * which would quietly delete most of the article the first time it was opened for editing.
 *
 * So blocks matching KEEP_VERBATIM become atomic nodes holding their original outerHTML.
 * They render exactly as they do in the reader view, can be selected and deleted as a
 * unit, and round-trip unchanged. Everything else — paragraphs, headings, lists, tables,
 * code — is normal editable content.
 */

import { Extension, Mark, Node, mergeAttributes } from '@tiptap/core';

/**
 * Selectors whose HTML must survive editing untouched.
 *
 * `pre` is here because ProseMirror code blocks hold plain text by definition, so the
 * hand-written syntax-colouring spans inside them cannot survive as marks — measured, 59
 * of 59 keyword spans were destroyed before this was added.
 */
export const KEEP_VERBATIM = [
  'figure',
  'pre',
  'nav.toc',
  'div.callout',
  'div.define',
  'div.worked',
  'div.budget',
  'div.part-head',
  'div.reading-demo',
];

export const RawBlock = Node.create({
  name: 'rawBlock',
  group: 'block',
  atom: true,
  selectable: true,
  draggable: true,

  addAttributes() {
    return {
      // rendered:false keeps these out of the serialized placeholder. Without it the
      // whole block's markup is emitted as an HTML attribute, whose quotes and angle
      // brackets then break the placeholder substitution in serializeWithRaw.
      html: { default: '', rendered: false },
      /** Shown in the editor chrome so a collapsed block is still identifiable. */
      label: { default: 'block', rendered: false },
    };
  },

  parseHTML() {
    // One rule per tag name, with the class check done in getAttrs. ProseMirror's `tag`
    // matcher does not reliably handle compound selectors like "div.callout" — figures
    // matched because "figure" is a bare tag, while every div-based block silently fell
    // through to the default parser and lost its wrapper.
    const tags = [...new Set(KEEP_VERBATIM.map((s) => s.split('.')[0]!))];
    return tags.map((tag) => ({
      tag,
      // Beat the built-in table and paragraph parsers, which would otherwise claim the
      // contents of a figure and strip its SVG.
      priority: 200,
      getAttrs: (element: HTMLElement) => {
        if (!KEEP_VERBATIM.some((selector) => element.matches(selector))) return false;
        return { html: element.outerHTML, label: describe(element) };
      },
    }));
  },

  renderHTML() {
    // The real markup is spliced back in by serializeWithRaw; this is only a placeholder
    // for the DOM serializer to hang the substitution on.
    return ['div', mergeAttributes({ 'data-raw': 'true' })];
  },

  addNodeView() {
    return ({ node }) => {
      const wrap = document.createElement('div');
      wrap.className = 'rawblock';
      wrap.contentEditable = 'false';

      const tag = document.createElement('span');
      tag.className = 'rawblock__tag';
      tag.textContent = node.attrs.label;

      const body = document.createElement('div');
      body.className = 'rawblock__body';
      body.innerHTML = node.attrs.html;

      wrap.append(tag, body);
      return { dom: wrap };
    };
  },
});

function describe(element: HTMLElement): string {
  if (element.tagName === 'FIGURE') {
    const label = element.querySelector('.fig-label')?.textContent?.trim();
    return label ? label.toLowerCase() : 'figure';
  }
  if (element.classList.contains('callout')) {
    return `callout · ${element.querySelector('.tag')?.textContent?.trim() ?? ''}`.trim();
  }
  if (element.classList.contains('define')) {
    return `definition · ${element.querySelector('.term')?.textContent?.trim() ?? ''}`.trim();
  }
  if (element.classList.contains('worked')) {
    return `worked example · ${element.querySelector('.tag')?.textContent?.trim() ?? ''}`.trim();
  }
  if (element.classList.contains('budget')) return 'parameter budget';
  if (element.tagName === 'NAV') return 'table of contents';
  return element.tagName.toLowerCase();
}

/**
 * Serialize the document, substituting each rawBlock's stored HTML.
 *
 * TipTap's getHTML() runs the DOM serializer, which cannot reproduce the original markup
 * from an atom node. Walking the document ourselves and splicing the stored strings back
 * in is what makes the round-trip lossless.
 */
export function serializeWithRaw(editor: { getJSON: () => any; getHTML: () => string }): string {
  const html = editor.getHTML();
  const raws: string[] = [];
  const walk = (node: any) => {
    if (node.type === 'rawBlock') raws.push(node.attrs.html);
    (node.content ?? []).forEach(walk);
  };
  walk(editor.getJSON());

  let i = 0;
  return html.replace(/<div[^>]*data-raw="true"[^>]*><\/div>/g, () => raws[i++] ?? '');
}


/**
 * Keep `class` and `id` on ordinary block elements.
 *
 * Without this, `<p class="filename">` and `<h2 id="s7">` lose their attributes on save:
 * the styling disappears and every in-page anchor in the table of contents breaks.
 */
export const KeepAttributes = Extension.create({
  name: 'keepAttributes',
  addGlobalAttributes() {
    const passthrough = (name: string) => ({
      default: null,
      parseHTML: (element: HTMLElement) => element.getAttribute(name),
      renderHTML: (attrs: Record<string, unknown>) =>
        attrs[name] ? { [name]: attrs[name] } : {},
    });
    return [
      {
        types: ['paragraph', 'heading', 'bulletList', 'orderedList', 'listItem', 'blockquote'],
        attributes: { class: passthrough('class'), id: passthrough('id') },
      },
    ];
  },
});

/**
 * Preserve inline `<span class="...">`.
 *
 * Section numbers live in `<span class="num">` inside each heading. StarterKit has no span
 * mark, so all 19 were flattened into the heading text — turning "01 The problem" into
 * "01The problem" and losing the styling that puts the number on its own line.
 */
export const PreservedSpan = Mark.create({
  name: 'preservedSpan',
  priority: 90,
  addAttributes() {
    return {
      class: {
        default: null,
        parseHTML: (element: HTMLElement) => element.getAttribute('class'),
        renderHTML: (attrs) => (attrs.class ? { class: attrs.class } : {}),
      },
    };
  },
  parseHTML() {
    return [{ tag: 'span' }];
  },
  renderHTML({ HTMLAttributes }) {
    return ['span', HTMLAttributes, 0];
  },
});
