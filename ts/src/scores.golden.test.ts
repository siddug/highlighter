/**
 * End-to-end conformance: the browser must score paragraphs exactly as PyTorch does.
 *
 * This is the test that matters most. It exercises the whole chain at once — tokenizer,
 * feature packing, base-63 weight decoding, embedding bag, local window, bidirectional
 * Blelloch scan, and head — against scores computed by the Python model on the same text.
 * Every other conformance test checks one link; this one checks the rope.
 *
 * Fixtures come from `python3 -m salience.train.export`.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { score } from './cpu';
import { WORD } from './features';
import { decodeWeights, type WeightFile } from './weights';

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (p: string) => readFileSync(resolve(HERE, p), 'utf8');

const weightFile = JSON.parse(read('../../weights/model.json')) as WeightFile;
const fixtures = read('../../shared/golden/scores.jsonl')
  .split('\n')
  .filter((l) => l !== '')
  .map((l) => JSON.parse(l) as { text: string; scores: number[] });

describe('end-to-end conformance with PyTorch', () => {
  const weights = decodeWeights(weightFile);

  it('loads a weight file matching this build', () => {
    expect(weightFile.paramCount).toBe(39361);
    expect(fixtures.length).toBeGreaterThan(20);
  });

  it('rejects weights from a different spec version', () => {
    expect(() => decodeWeights({ ...weightFile, specVersion: 'v0' })).toThrow(/spec/);
  });

  it('rejects a truncated tensor', () => {
    const broken = structuredClone(weightFile);
    broken.tensors.headOutB!.data = '';
    expect(() => decodeWeights(broken)).toThrow(/expected 1 parameters/);
  });

  it('produces the same score for every word', () => {
    let worst = 0;
    let worstText = '';

    for (const { text, scores } of fixtures) {
      const got = score(text, weights)
        .filter((t) => t.cls === WORD)
        .map((t) => t.score);

      expect(got).toHaveLength(scores.length);
      for (let i = 0; i < got.length; i++) {
        const diff = Math.abs(got[i]! - scores[i]!);
        if (diff > worst) {
          worst = diff;
          worstText = text.slice(0, 60);
        }
      }
    }

    // float32 accumulation order differs between the two implementations, so exact
    // equality is not available. Anything above this is a structural divergence.
    expect(worst, `worst on: ${worstText}`).toBeLessThan(2e-4);
  });

  it('ranks words consistently with PyTorch', () => {
    // Scores could agree numerically yet order differently near ties; the product is a
    // ranking, so check the ranking directly.
    for (const { text, scores } of fixtures) {
      const got = score(text, weights)
        .filter((t) => t.cls === WORD)
        .map((t) => t.score);
      const k = Math.max(1, Math.round(0.2 * got.length));
      const topOf = (xs: number[]) =>
        new Set(
          xs
            .map((v, i) => ({ v, i }))
            .sort((a, b) => b.v - a.v)
            .slice(0, k)
            .map((x) => x.i),
        );
      const a = topOf(got);
      const b = topOf(scores);
      const overlap = [...a].filter((i) => b.has(i)).length;
      expect(overlap / k).toBeGreaterThanOrEqual(0.95);
    }
  });
});
