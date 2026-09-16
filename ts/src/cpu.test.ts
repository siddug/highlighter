import { describe, expect, it } from 'vitest';
import { WORD } from './features';
import { D, LEVELS, PARAM_COUNT, mulberry32, randomWeights } from './model';
import { blelloch, forward, score } from './cpu';

describe('parameter budget', () => {
  it('stays under 40k, the gpu-lexer-scale target', () => {
    expect(PARAM_COUNT).toBeLessThan(40_000);
    expect(PARAM_COUNT).toBeGreaterThan(30_000);
  });
});

describe('blelloch scan', () => {
  /**
   * With every level gate saturated to 1, the combine becomes truly associative and
   * must reproduce the sequential recurrence s_i = z_i * s_{i-1} + w_i exactly.
   * This validates the up-sweep/down-sweep plumbing independently of the gate params.
   */
  function sequential(z: Float32Array, w: Float32Array, n: number): Float32Array {
    const out = new Float32Array(n * D);
    const state = new Float32Array(D);
    for (let t = 0; t < n; t++) {
      for (let d = 0; d < D; d++) {
        state[d] = z[t * D + d]! * state[d]! + w[t * D + d]!;
        out[t * D + d] = state[d]!;
      }
    }
    return out;
  }

  const saturated = new Float32Array(LEVELS * D).fill(30); // sigmoid(30) ~ 1

  for (const n of [1, 2, 3, 5, 8, 9, 16, 17, 33, 64]) {
    it(`matches the sequential recurrence at n=${n}`, () => {
      const rand = mulberry32(n * 7919);
      const z = new Float32Array(n * D);
      const w = new Float32Array(n * D);
      for (let i = 0; i < n * D; i++) {
        z[i] = rand() * 0.9 + 0.05;
        w[i] = rand() * 2 - 1;
      }

      const fast = blelloch(z, w, n, saturated, saturated);
      const slow = sequential(z, w, n);

      for (let i = 0; i < n * D; i++) {
        expect(fast[i]!).toBeCloseTo(slow[i]!, 4);
      }
    });
  }

  /**
   * The architectural claim: with the forget gate open (z=1) a single leaf's
   * contribution reaches every later position, however far. This is what gives the
   * model a whole-document receptive field in log depth.
   *
   * Note the flip side, which matters at training time: when z < 1 the carry decays
   * geometrically, and over ~100 tokens with random init it underflows float32 to
   * exactly zero. Training will need z biased open at init or long-range context is
   * unlearnable.
   */
  it('propagates a single leaf across the whole sequence when the gate is open', () => {
    const n = 512;
    const z = new Float32Array(n * D).fill(1);
    const w = new Float32Array(n * D);
    for (let d = 0; d < D; d++) w[d] = 1; // only leaf 0 carries signal

    const out = blelloch(z, w, n, saturated, saturated);
    for (const i of [0, 1, 7, 63, 255, n - 1]) {
      expect(out[i * D]!).toBeCloseTo(1, 5);
    }
  });

  it('is unaffected by the padding tail', () => {
    // Real leaves are identical; only the padded length differs (8 vs 16).
    const rand = mulberry32(42);
    const make = (n: number) => {
      const z = new Float32Array(n * D);
      const w = new Float32Array(n * D);
      const r = mulberry32(42);
      for (let i = 0; i < n * D; i++) {
        z[i] = r() * 0.9 + 0.05;
        w[i] = r() * 2 - 1;
      }
      return { z, w };
    };
    rand();

    const a = make(8);
    const b = make(16);
    const ra = blelloch(a.z, a.w, 8, saturated, saturated);
    const rb = blelloch(b.z, b.w, 8, saturated, saturated); // n=8 but buffers sized 16

    for (let i = 0; i < 8 * D; i++) expect(ra[i]!).toBeCloseTo(rb[i]!, 5);
  });
});

describe('forward pass', () => {
  const w = randomWeights(1);

  it('produces a score in [0,1] for every word', () => {
    const tokens = score('The mitochondria is the powerhouse of the cell.', w);
    expect(tokens.length).toBeGreaterThan(0);
    for (const t of tokens) {
      expect(t.score).toBeGreaterThanOrEqual(0);
      expect(t.score).toBeLessThanOrEqual(1);
    }
  });

  it('scores non-word tokens as exactly zero', () => {
    for (const t of score('Hello, world! Again.', w)) {
      if (t.cls !== WORD) expect(t.score).toBe(0);
    }
  });

  it('reconstructs the input from token spans', () => {
    const text = 'Alpha beta gamma.\n\nDelta epsilon.';
    expect(score(text, w).map((t) => t.text).join('')).toBe(text);
  });

  it('handles empty and whitespace-only input', () => {
    expect(score('', w)).toEqual([]);
    expect(score('   ', w).every((t) => t.score === 0)).toBe(true);
  });

  it('is deterministic', () => {
    const a = score('Repeatable output please.', w);
    const b = score('Repeatable output please.', w);
    expect(a).toEqual(b);
  });

  /**
   * A word's score must depend on context well outside its five-token window.
   * Here "The" at index 0 is moved by words ~8 tokens downstream.
   */
  it('lets context beyond the local window change a score', () => {
    const head = 'The reactor achieved criticality. ';
    const a = score(head + 'Nothing unusual happened here.', w);
    const b = score(head + 'Catastrophic meltdown destroyed everything.', w);

    const firstWordA = a.find((t) => t.cls === WORD)!;
    const firstWordB = b.find((t) => t.cls === WORD)!;
    expect(firstWordA.text).toBe(firstWordB.text);
    expect(Math.abs(firstWordA.score - firstWordB.score)).toBeGreaterThan(1e-6);
  });

  it('exposes intermediates with the expected shapes', () => {
    const { inter } = forward('Two words.', w);
    const n = inter.feats.length;
    expect(inter.emb).toHaveLength(n * D);
    expect(inter.local).toHaveLength(n * D);
    expect(inter.gf).toHaveLength(n * D);
    expect(inter.gb).toHaveLength(n * D);
    expect(Array.from(inter.local).every(Number.isFinite)).toBe(true);
    expect(Array.from(inter.gf).every(Number.isFinite)).toBe(true);
  });

  it('survives a long document', () => {
    const long = 'The quick brown fox jumps over the lazy dog. '.repeat(120);
    const tokens = score(long, w);
    expect(tokens.length).toBeGreaterThan(1000);
    expect(tokens.every((t) => Number.isFinite(t.score))).toBe(true);
  });
});
