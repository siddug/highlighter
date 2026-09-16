/**
 * Decode the base-63 wire format produced by salience/train/export.py.
 *
 * One character per parameter, zigzag-encoded, with a float32 scale per tensor. This is
 * what makes the model small enough to ship: 39,361 parameters in 39,361 characters.
 */

import { SPEC_VERSION } from './features';
import { PARAM_COUNT, TENSOR_SHAPES, type Weights } from './model';

const ALPHABET = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

/** Character code -> zigzag value, built once. */
const DECODE = new Int8Array(128).fill(-1);
for (let i = 0; i < ALPHABET.length; i++) DECODE[ALPHABET.charCodeAt(i)] = i;

export interface WeightFile {
  specVersion: string;
  paramCount: number;
  alphabet: string;
  tensors: Record<string, { shape: number[]; scale: number; data: string }>;
}

function decodeTensor(codes: string, scale: number, expected: number): Float32Array {
  if (codes.length !== expected) {
    throw new Error(`expected ${expected} parameters, got ${codes.length}`);
  }
  const out = new Float32Array(expected);
  for (let i = 0; i < expected; i++) {
    const u = DECODE[codes.charCodeAt(i)]!;
    if (u < 0) throw new Error(`bad character ${JSON.stringify(codes[i])} at ${i}`);
    // Zigzag: 0,1,2,3,4 -> 0,-1,1,-2,2
    out[i] = ((u >> 1) ^ -(u & 1)) * scale;
  }
  return out;
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

export function decodeWeights(file: WeightFile): Weights {
  if (file.specVersion !== SPEC_VERSION) {
    throw new Error(
      `weights were trained against spec ${file.specVersion}, this build is ${SPEC_VERSION}`,
    );
  }
  if (file.paramCount !== PARAM_COUNT) {
    throw new Error(`weights have ${file.paramCount} parameters, model expects ${PARAM_COUNT}`);
  }

  const out: Record<string, unknown> = { version: file.specVersion };
  for (const { path, size } of TENSOR_SHAPES) {
    const tensor = file.tensors[path];
    if (!tensor) throw new Error(`weight file is missing tensor ${path}`);
    setPath(out, path, decodeTensor(tensor.data, tensor.scale, size));
  }
  return out as unknown as Weights;
}
