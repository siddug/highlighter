/**
 * The portable schema — deliberately small.
 *
 * The article is written here but read elsewhere: the end state is pasting it into a blog
 * that understands TipTap and images and nothing else. So the schema is restricted to
 * elements that survive that trip, and `scripts/portablize.py` rewrote the article to use
 * only those.
 *
 * An earlier version preserved bespoke blocks verbatim as atomic nodes. That kept the
 * document pretty here and unusable anywhere else, and it left callouts and definitions
 * uneditable. Converting them to blockquotes loses a little styling and gains both
 * editability and portability.
 */

import { Extension, type Extensions } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import { TableKit } from '@tiptap/extension-table';
import Image from '@tiptap/extension-image';
import Highlight from '@tiptap/extension-highlight';

/**
 * Keep `id` on headings.
 *
 * `id` is standard HTML and harmless anywhere, but ProseMirror drops unknown attributes,
 * so without this the article's 19 heading anchors disappear on the first save — the
 * reader view would silently lose every deep link the moment anyone edited a word.
 */
const HeadingAnchors = Extension.create({
  name: 'headingAnchors',
  addGlobalAttributes() {
    return [
      {
        types: ['heading'],
        attributes: {
          id: {
            default: null,
            parseHTML: (element: HTMLElement) => element.getAttribute('id'),
            renderHTML: (attrs) => (attrs.id ? { id: attrs.id } : {}),
          },
        },
      },
    ];
  },
});

export const extensions: Extensions = [
  StarterKit.configure({
    heading: { levels: [2, 3, 4] },
  }),
  TableKit.configure({ table: { resizable: false } }),
  Image.configure({ inline: false, allowBase64: false }),
  // <mark> carries the highlighted words in Figure 1, which is the article's whole subject.
  Highlight,
  HeadingAnchors,
];
