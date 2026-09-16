/**
 * Implementation of shared/spec/features.md v2.
 *
 * salience/features.py is the other implementation. shared/golden/ proves they agree.
 * If you change anything here, change the spec first.
 */

import stopwordsRaw from '../../shared/spec/stopwords.txt?raw';

export const SPEC_VERSION = 'v2';

export const WORD = 0;
export const SPACE = 1;
export const NEWLINE = 2;
export const PUNCT = 3;

/** Row map from spec section 4. Offsets into the single 737-row embedding table. */
export const ROWS = {
  CLASS: 0,
  HASH_A: 8,
  HASH_B: 264,
  SUFFIX: 520,
  PREFIX: 584,
  STOPWORD: 616,
  FLAGS: 632,
  LENGTH: 641,
  SENT_POS: 657,
  DOC_POS: 689,
  DOC_FREQ: 721,
  TOTAL: 737,
} as const;

export const FLAG = {
  CAPITALIZED: 0,
  ALL_CAPS: 1,
  HAS_DIGIT: 2,
  ALL_DIGITS: 3,
  HAS_HYPHEN: 4,
  HAS_APOSTROPHE: 5,
  SENTENCE_START: 6,
  PARAGRAPH_START: 7,
  BEFORE_SENTENCE_END: 8,
} as const;

export interface TokenFeatures {
  cls: number;
  /** Start offset in code points into the normalized text. */
  start: number;
  /** End offset (exclusive) in code points. */
  end: number;
  text: string;
  /** Sorted embedding row indices; the token embedding is the sum of these rows. */
  rows: number[];
  /** Token index of the first WORD of this token's sentence, or -1. */
  sentenceFirstWord: number;
  /** Token index of the last WORD of the previous sentence, or -1. */
  prevSentenceLastWord: number;
}

// --- Determinism primitives (spec section 1) -------------------------------

const LETTER_RE = /[\p{L}\p{M}]/u;
const DIGIT_RE = /\p{N}/u;
const UPPER_RE = /\p{Lu}/u;

function isLetter(cp: string): boolean {
  return LETTER_RE.test(cp);
}
function isDigit(cp: string): boolean {
  return DIGIT_RE.test(cp);
}
function isAlnum(cp: string): boolean {
  return isLetter(cp) || isDigit(cp);
}

/** ASCII-only lowercasing. Never use toLowerCase() — it diverges from Python on some locales. */
export function asciiLower(s: string): string {
  let out = '';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    out += c >= 0x41 && c <= 0x5a ? String.fromCharCode(c + 32) : s[i]!;
  }
  return out;
}

const encoder = new TextEncoder();

function fnv(bytes: Uint8Array, basis: number, prime: number): number {
  let h = basis >>> 0;
  for (let i = 0; i < bytes.length; i++) {
    h = (h ^ bytes[i]!) >>> 0;
    h = Math.imul(h, prime) >>> 0;
  }
  return h >>> 0;
}

function mix(h0: number): number {
  let h = h0 >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, 2246822507) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 3266489909) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h >>> 0;
}

/** Normalize line endings. Must run before tokenizing (spec section 1.5). */
export function normalizeNewlines(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

// --- Stopwords (spec section 3.2) ------------------------------------------

const STOPWORD_RANK = new Map<string, number>();
{
  let rank = 0;
  for (const line of stopwordsRaw.split('\n')) {
    const w = line.trim();
    if (w === '' || w.startsWith('#')) continue;
    if (!STOPWORD_RANK.has(w)) STOPWORD_RANK.set(w, rank);
    rank++;
  }
}

function stopwordBucket(lower: string): number {
  const rank = STOPWORD_RANK.get(lower);
  return rank === undefined ? 0 : Math.min(15, 1 + Math.floor(rank / 16));
}

// --- Buckets (spec section 3.4) --------------------------------------------

const LENGTH_EDGES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 19, 24, 31, 47];
const FREQ_EDGES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64];

function bucketize(value: number, edges: number[]): number {
  for (let i = 0; i < edges.length; i++) {
    if (value <= edges[i]!) return i;
  }
  return edges.length;
}

// --- Tokenization (spec section 2) -----------------------------------------

interface RawToken {
  cls: number;
  start: number;
  end: number;
  text: string;
}

const SENTENCE_END_CHARS = new Set(['.', '!', '?', '…']);
const CLOSING_CHARS = new Set(['"', "'", ')', ']', '}', '»', '”', '’']);

/**
 * Split into tokens. Every code point lands in exactly one token, so concatenating
 * all token texts reproduces the normalized input.
 */
export function tokenize(normalized: string): RawToken[] {
  const cps = Array.from(normalized);
  const n = cps.length;
  const tokens: RawToken[] = [];
  let i = 0;

  const push = (cls: number, start: number, end: number) => {
    tokens.push({ cls, start, end, text: cps.slice(start, end).join('') });
  };

  while (i < n) {
    const ch = cps[i]!;

    if (ch === '\n') {
      push(NEWLINE, i, i + 1);
      i += 1;
    } else if (ch === ' ' || ch === '\t') {
      const start = i;
      while (i < n && (cps[i] === ' ' || cps[i] === '\t')) i++;
      push(SPACE, start, i);
    } else if (isAlnum(ch)) {
      const start = i;
      i++;
      while (i < n) {
        const c = cps[i]!;
        if (isAlnum(c)) {
          i++;
        } else if ((c === "'" || c === '-') && i + 1 < n && isAlnum(cps[i + 1]!)) {
          // Internal only: the previous code point is alphanumeric by construction.
          i++;
        } else if (
          (c === '.' || c === ',') &&
          i + 1 < n &&
          isDigit(cps[i + 1]!) &&
          isDigit(cps[i - 1]!)
        ) {
          // Keep numbers whole: 1.6 and 10,000 are single facts.
          i++;
        } else {
          break;
        }
      }
      push(WORD, start, i);
    } else {
      push(PUNCT, i, i + 1);
      i += 1;
    }
  }

  return tokens;
}

/** Mark the last token of each sentence (spec section 2, "Sentence segmentation"). */
function markSentenceEnds(tokens: RawToken[]): boolean[] {
  const n = tokens.length;
  const ends = new Array<boolean>(n).fill(false);

  for (let t = 0; t < n; t++) {
    const tok = tokens[t]!;

    if (tok.cls === NEWLINE) {
      ends[t] = true;
      continue;
    }
    if (tok.cls !== PUNCT || !SENTENCE_END_CHARS.has(tok.text)) continue;

    // Absorb trailing closing punctuation into this sentence.
    let j = t + 1;
    while (j < n && tokens[j]!.cls === PUNCT && CLOSING_CHARS.has(tokens[j]!.text)) j++;
    const lastIdx = j - 1;

    let k = j;
    while (k < n && tokens[k]!.cls === SPACE) k++;

    if (k >= n) {
      ends[lastIdx] = true;
    } else {
      const next = tokens[k]!;
      if (next.cls === WORD) {
        const first = Array.from(next.text)[0]!;
        if (UPPER_RE.test(first) || isDigit(first)) ends[lastIdx] = true;
      }
    }
  }

  return ends;
}

// --- Feature extraction ----------------------------------------------------

/** Tokenize `text` and compute the embedding row indices for every token. */
export function extractFeatures(text: string): TokenFeatures[] {
  const normalized = normalizeNewlines(text);
  const tokens = tokenize(normalized);
  const n = tokens.length;
  if (n === 0) return [];

  const lower = tokens.map((t) => (t.cls === WORD ? asciiLower(t.text) : t.text));

  // Sentence ids.
  const ends = markSentenceEnds(tokens);
  const sid = new Int32Array(n);
  {
    let cur = 0;
    for (let t = 0; t < n; t++) {
      sid[t] = cur;
      if (ends[t]) cur++;
    }
  }

  // Position within sentence.
  const sentCount = new Map<number, number>();
  const sentIndex = new Int32Array(n);
  for (let t = 0; t < n; t++) {
    const s = sid[t]!;
    const c = sentCount.get(s) ?? 0;
    sentIndex[t] = c;
    sentCount.set(s, c + 1);
  }

  // Pointers: first WORD of this sentence, last WORD of the previous sentence.
  const firstWordOfSent = new Map<number, number>();
  const lastWordOfSent = new Map<number, number>();
  for (let t = 0; t < n; t++) {
    if (tokens[t]!.cls !== WORD) continue;
    const s = sid[t]!;
    if (!firstWordOfSent.has(s)) firstWordOfSent.set(s, t);
    lastWordOfSent.set(s, t);
  }

  // Within-document frequency of each lowercased word form.
  const docFreq = new Map<string, number>();
  for (let t = 0; t < n; t++) {
    if (tokens[t]!.cls !== WORD) continue;
    const key = lower[t]!;
    docFreq.set(key, (docFreq.get(key) ?? 0) + 1);
  }

  // Paragraph starts: first WORD of the document, and the first WORD after >=2 newlines.
  const paragraphStart = new Array<boolean>(n).fill(false);
  {
    let pending = true; // document start counts as a paragraph break
    let consecutiveNewlines = 0;
    for (let t = 0; t < n; t++) {
      const cls = tokens[t]!.cls;
      if (cls === NEWLINE) {
        consecutiveNewlines++;
        if (consecutiveNewlines >= 2) pending = true;
      } else if (cls === WORD) {
        if (pending) {
          paragraphStart[t] = true;
          pending = false;
        }
        consecutiveNewlines = 0;
      } else if (cls === PUNCT) {
        consecutiveNewlines = 0;
      }
      // SPACE neither breaks a newline run nor consumes a pending break.
    }
  }

  const out: TokenFeatures[] = [];

  for (let t = 0; t < n; t++) {
    const tok = tokens[t]!;
    const s = sid[t]!;
    const rows: number[] = [];

    rows.push(ROWS.CLASS + tok.cls);

    const sentLen = sentCount.get(s) ?? 1;
    rows.push(ROWS.SENT_POS + Math.min(31, Math.floor((32 * sentIndex[t]!) / Math.max(1, sentLen))));
    rows.push(ROWS.DOC_POS + Math.min(31, Math.floor((32 * t) / Math.max(1, n))));

    let flags = 0;

    if (tok.cls === WORD) {
      const lw = lower[t]!;
      const bytes = encoder.encode(lw);
      const suffixBytes = bytes.subarray(Math.max(0, bytes.length - 3));
      const prefixBytes = bytes.subarray(0, Math.min(3, bytes.length));

      rows.push(ROWS.HASH_A + (mix(fnv(bytes, 2166136261, 16777619)) & 0xff));
      rows.push(ROWS.HASH_B + (mix(fnv(bytes, 1166136321, 2246822519)) & 0xff));
      rows.push(ROWS.SUFFIX + (mix(fnv(suffixBytes, 2166136261, 16777619)) & 0x3f));
      rows.push(ROWS.PREFIX + (mix(fnv(prefixBytes, 2166136261, 16777619)) & 0x1f));
      rows.push(ROWS.STOPWORD + stopwordBucket(lw));

      const cps = Array.from(tok.text);
      rows.push(ROWS.LENGTH + bucketize(cps.length, LENGTH_EDGES));
      rows.push(ROWS.DOC_FREQ + bucketize(docFreq.get(lw) ?? 1, FREQ_EDGES));

      let letters = 0;
      let uppers = 0;
      let digits = 0;
      let nonDigits = 0;
      for (const c of cps) {
        if (isLetter(c)) {
          letters++;
          if (UPPER_RE.test(c)) uppers++;
        }
        if (isDigit(c)) digits++;
        else nonDigits++;
      }

      const allCaps = letters >= 2 && uppers === letters;
      if (allCaps) flags |= 1 << FLAG.ALL_CAPS;
      else if (UPPER_RE.test(cps[0]!)) flags |= 1 << FLAG.CAPITALIZED;

      if (digits > 0) flags |= 1 << FLAG.HAS_DIGIT;
      if (nonDigits === 0) flags |= 1 << FLAG.ALL_DIGITS;
      if (tok.text.includes('-')) flags |= 1 << FLAG.HAS_HYPHEN;
      if (tok.text.includes("'")) flags |= 1 << FLAG.HAS_APOSTROPHE;
      if (firstWordOfSent.get(s) === t) flags |= 1 << FLAG.SENTENCE_START;
      if (paragraphStart[t]) flags |= 1 << FLAG.PARAGRAPH_START;

      let k = t + 1;
      while (k < n && tokens[k]!.cls === SPACE) k++;
      if (k < n && tokens[k]!.cls === PUNCT && SENTENCE_END_CHARS.has(tokens[k]!.text)) {
        flags |= 1 << FLAG.BEFORE_SENTENCE_END;
      }
    }

    for (let b = 0; b < 9; b++) {
      if (flags & (1 << b)) rows.push(ROWS.FLAGS + b);
    }

    rows.sort((a, b) => a - b);

    out.push({
      cls: tok.cls,
      start: tok.start,
      end: tok.end,
      text: tok.text,
      rows,
      sentenceFirstWord: firstWordOfSent.get(s) ?? -1,
      prevSentenceLastWord: s > 0 ? (lastWordOfSent.get(s - 1) ?? -1) : -1,
    });
  }

  return out;
}
