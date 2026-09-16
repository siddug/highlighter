import { describe, expect, it } from 'vitest';
import {
  FLAG,
  NEWLINE,
  PUNCT,
  ROWS,
  SPACE,
  WORD,
  asciiLower,
  extractFeatures,
  normalizeNewlines,
  tokenize,
} from './features';

const SAMPLES = [
  '',
  'hello',
  'Hello world.',
  "It's a state-of-the-art NASA probe, launched in 2019.",
  'First paragraph.\n\nSecond paragraph starts here.',
  'Windows\r\nline\rendings',
  'tabs\tand   spaces',
  'CJK 日本語 text and emoji 👨‍👩‍👧‍👦 here',
  '__dunder__ and 1234567890 and well- dangling',
  'Dr. Smith went to Washington. He arrived.',
  'a--b',
  'ellipsis… then more',
  'zero​width',
];

describe('tokenize', () => {
  it('covers every code point exactly once', () => {
    for (const s of SAMPLES) {
      const norm = normalizeNewlines(s);
      const tokens = tokenize(norm);
      expect(tokens.map((t) => t.text).join('')).toBe(norm);
      for (let i = 1; i < tokens.length; i++) {
        expect(tokens[i]!.start).toBe(tokens[i - 1]!.end);
      }
    }
  });

  it('keeps internal apostrophes and hyphens inside words', () => {
    const tokens = tokenize("it's state-of-the-art");
    const words = tokens.filter((t) => t.cls === WORD).map((t) => t.text);
    expect(words).toEqual(["it's", 'state-of-the-art']);
  });

  it('splits a dangling hyphen off the word', () => {
    const tokens = tokenize('well- done');
    expect(tokens.map((t) => [t.cls, t.text])).toEqual([
      [WORD, 'well'],
      [PUNCT, '-'],
      [SPACE, ' '],
      [WORD, 'done'],
    ]);
  });

  it('emits one token per newline', () => {
    const tokens = tokenize('a\n\nb');
    expect(tokens.filter((t) => t.cls === NEWLINE)).toHaveLength(2);
  });

  it('treats em dash as punctuation but hyphen-minus as internal', () => {
    expect(tokenize('a—b').filter((t) => t.cls === WORD).map((t) => t.text)).toEqual(['a', 'b']);
    expect(tokenize('a-b').filter((t) => t.cls === WORD).map((t) => t.text)).toEqual(['a-b']);
  });
});

describe('asciiLower', () => {
  it('lowercases ASCII only', () => {
    expect(asciiLower('HeLLo')).toBe('hello');
    expect(asciiLower('İSTANBUL')).toBe('İstanbul');
    expect(asciiLower('STRASSE')).toBe('strasse');
  });
});

describe('extractFeatures', () => {
  it('returns rows sorted, unique-ish, and in range', () => {
    for (const s of SAMPLES) {
      for (const f of extractFeatures(s)) {
        expect(f.rows).toEqual([...f.rows].sort((a, b) => a - b));
        for (const r of f.rows) {
          expect(r).toBeGreaterThanOrEqual(0);
          expect(r).toBeLessThan(ROWS.TOTAL);
        }
      }
    }
  });

  it('gives non-word tokens exactly three rows', () => {
    const feats = extractFeatures('hi there.');
    for (const f of feats) {
      if (f.cls !== WORD) expect(f.rows).toHaveLength(3);
    }
  });

  it('flags sentence and paragraph starts', () => {
    const feats = extractFeatures('One two.\n\nThree four.');
    const words = feats.filter((f) => f.cls === WORD);
    const hasFlag = (i: number, bit: number) => words[i]!.rows.includes(ROWS.FLAGS + bit);

    expect(hasFlag(0, FLAG.SENTENCE_START)).toBe(true);
    expect(hasFlag(0, FLAG.PARAGRAPH_START)).toBe(true);
    expect(hasFlag(1, FLAG.SENTENCE_START)).toBe(false);
    expect(hasFlag(2, FLAG.SENTENCE_START)).toBe(true);
    expect(hasFlag(2, FLAG.PARAGRAPH_START)).toBe(true);
    expect(hasFlag(3, FLAG.PARAGRAPH_START)).toBe(false);
  });

  it('flags the word before sentence-final punctuation', () => {
    const feats = extractFeatures('Alpha beta.');
    const words = feats.filter((f) => f.cls === WORD);
    expect(words[0]!.rows.includes(ROWS.FLAGS + FLAG.BEFORE_SENTENCE_END)).toBe(false);
    expect(words[1]!.rows.includes(ROWS.FLAGS + FLAG.BEFORE_SENTENCE_END)).toBe(true);
  });

  it('distinguishes capitalized from all-caps', () => {
    const feats = extractFeatures('Apple NASA lower');
    const words = feats.filter((f) => f.cls === WORD);
    expect(words[0]!.rows.includes(ROWS.FLAGS + FLAG.CAPITALIZED)).toBe(true);
    expect(words[0]!.rows.includes(ROWS.FLAGS + FLAG.ALL_CAPS)).toBe(false);
    expect(words[1]!.rows.includes(ROWS.FLAGS + FLAG.ALL_CAPS)).toBe(true);
    expect(words[1]!.rows.includes(ROWS.FLAGS + FLAG.CAPITALIZED)).toBe(false);
    expect(words[2]!.rows.includes(ROWS.FLAGS + FLAG.CAPITALIZED)).toBe(false);
  });

  it('assigns stopwords a nonzero bucket and content words zero', () => {
    const feats = extractFeatures('the photosynthesis');
    const words = feats.filter((f) => f.cls === WORD);
    const bucket = (i: number) =>
      words[i]!.rows.find((r) => r >= ROWS.STOPWORD && r < ROWS.FLAGS)! - ROWS.STOPWORD;
    expect(bucket(0)).toBe(1);
    expect(bucket(1)).toBe(0);
  });

  it('counts within-document frequency', () => {
    const feats = extractFeatures('cat dog cat Cat');
    const words = feats.filter((f) => f.cls === WORD);
    const freq = (i: number) =>
      words[i]!.rows.find((r) => r >= ROWS.DOC_FREQ && r < ROWS.TOTAL)! - ROWS.DOC_FREQ;
    // "cat" appears 3 times (case-insensitive) -> bucket 2; "dog" once -> bucket 0.
    expect(freq(0)).toBe(2);
    expect(freq(1)).toBe(0);
    expect(freq(3)).toBe(2);
  });

  it('points at the first word of the sentence and the last of the previous', () => {
    const feats = extractFeatures('Alpha beta. Gamma delta.');
    // tokens: Alpha(0) ' '(1) beta(2) .(3) ' '(4) Gamma(5) ' '(6) delta(7) .(8)
    expect(feats[0]!.sentenceFirstWord).toBe(0);
    expect(feats[0]!.prevSentenceLastWord).toBe(-1);
    expect(feats[5]!.sentenceFirstWord).toBe(5);
    expect(feats[5]!.prevSentenceLastWord).toBe(2);
  });

  it('handles empty input', () => {
    expect(extractFeatures('')).toEqual([]);
  });

  it('is stable across repeated calls', () => {
    for (const s of SAMPLES) {
      expect(extractFeatures(s)).toEqual(extractFeatures(s));
    }
  });
});

describe('normalizeNewlines', () => {
  it('collapses CRLF and lone CR', () => {
    expect(normalizeNewlines('a\r\nb\rc\nd')).toBe('a\nb\nc\nd');
  });
});

describe('spacing', () => {
  it('merges runs of spaces and tabs into one token', () => {
    const tokens = tokenize('a \t  b');
    expect(tokens.map((t) => t.cls)).toEqual([WORD, SPACE, WORD]);
  });
});
