/**
 * Model dimensions and weight container.
 *
 * Shapes are the contract between py/train/export.py and ts/src/cpu.ts. Changing any
 * of them is a breaking change: bump SPEC_VERSION in features.ts and retrain.
 */

import { ROWS } from './features';

export const D = 32; // d_model
export const WINDOW = 5; // local context, +/- 2
export const PTRS = 2; // sentence-start and previous-sentence-end pointers
export const LEVELS = 12; // scan tree levels, indexed min(level, 11)
export const GATE = 16; // head gate units
export const HID = 72; // head hidden units

/** Head input: local (32) + forward global (32) + backward global (32). */
export const HEAD_IN = D * 3;

export interface Weights {
  version: string;
  /** Embedding table, ROWS.TOTAL x D, row-major. */
  emb: Float32Array;

  localBias: Float32Array; // D
  localWin: Float32Array; // WINDOW x D, elementwise per position
  localPtr: Float32Array; // PTRS x D, elementwise

  fwd: ScanWeights;
  bwd: ScanWeights;

  headGateW: Float32Array; // GATE x HEAD_IN
  headGateB: Float32Array; // GATE
  headHidW: Float32Array; // HID x (HEAD_IN + GATE)
  headHidB: Float32Array; // HID
  headOutW: Float32Array; // 1 x HID
  headOutB: Float32Array; // 1
}

export interface ScanWeights {
  wz: Float32Array; // D x D
  bz: Float32Array; // D
  wc: Float32Array; // D x D
  bc: Float32Array; // D
  up: Float32Array; // LEVELS x D
  down: Float32Array; // LEVELS x D
}

interface Shape {
  path: string;
  size: number;
}

/** Every tensor in declaration order, used by the exporter and the decoder. */
export const TENSOR_SHAPES: Shape[] = [
  { path: 'emb', size: ROWS.TOTAL * D },
  { path: 'localBias', size: D },
  { path: 'localWin', size: WINDOW * D },
  { path: 'localPtr', size: PTRS * D },
  ...(['fwd', 'bwd'] as const).flatMap((dir) => [
    { path: `${dir}.wz`, size: D * D },
    { path: `${dir}.bz`, size: D },
    { path: `${dir}.wc`, size: D * D },
    { path: `${dir}.bc`, size: D },
    { path: `${dir}.up`, size: LEVELS * D },
    { path: `${dir}.down`, size: LEVELS * D },
  ]),
  { path: 'headGateW', size: GATE * HEAD_IN },
  { path: 'headGateB', size: GATE },
  { path: 'headHidW', size: HID * (HEAD_IN + GATE) },
  { path: 'headHidB', size: HID },
  { path: 'headOutW', size: HID },
  { path: 'headOutB', size: 1 },
];

export const PARAM_COUNT = TENSOR_SHAPES.reduce((n, s) => n + s.size, 0);

/** Deterministic PRNG so random-weight runs are reproducible across languages. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function setPath(target: Record<string, unknown>, path: string, value: Float32Array): void {
  const parts = path.split('.');
  let node = target;
  for (let i = 0; i < parts.length - 1; i++) {
    const key = parts[i]!;
    node[key] ??= {};
    node = node[key] as Record<string, unknown>;
  }
  node[parts[parts.length - 1]!] = value;
}

/**
 * Initial bias on the scan's forget gate, pushing sigmoid(z) to ~0.88 at init.
 *
 * Without this the gate sits near 0.5 and the carry decays as 0.5^distance, which
 * underflows float32 within ~100 tokens — the model would start with no long-range
 * context at all and have no gradient with which to learn any. Training must use the
 * same initialization.
 */
export const FORGET_BIAS_INIT = 2.0;

/**
 * Initial value for the scan's per-level carry gate, sigmoid(3.0) ~ 0.95.
 *
 * Same failure mode as FORGET_BIAS_INIT, one level up: at zero the gate sits at 0.5 and the
 * carry loses half its magnitude at every level of the tree.
 */
export const LEVEL_GATE_INIT = 3.0;

/**
 * Untrained weights for the thin slice. Output is meaningless; the point is to
 * exercise the full pipeline end to end.
 */
export function randomWeights(seed = 1): Weights {
  const rand = mulberry32(seed);
  const out: Record<string, unknown> = { version: 'random' };
  for (const { path, size } of TENSOR_SHAPES) {
    const arr = new Float32Array(size);
    // Scaled so activations land in a sane range before any training.
    const scale = 1 / Math.sqrt(size / D);
    for (let i = 0; i < size; i++) arr[i] = (rand() * 2 - 1) * scale;
    if (path.endsWith('.bz')) {
      for (let i = 0; i < size; i++) arr[i] = arr[i]! + FORGET_BIAS_INIT;
    }
    if (path.endsWith('.up') || path.endsWith('.down')) {
      for (let i = 0; i < size; i++) arr[i] = arr[i]! + LEVEL_GATE_INIT;
    }
    setPath(out, path, arr);
  }
  return out as unknown as Weights;
}
