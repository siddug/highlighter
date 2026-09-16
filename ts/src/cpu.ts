/**
 * CPU reference implementation of the forward pass.
 *
 * This is the correctness oracle. The WGSL port in ts/src/wgsl/ must agree with it,
 * and py/train/model.py must agree with it. Clarity beats speed here.
 */

import { WORD, extractFeatures, type TokenFeatures } from './features';
import { D, GATE, HEAD_IN, HID, LEVELS, PTRS, WINDOW, type ScanWeights, type Weights } from './model';

export interface ScoredToken {
  text: string;
  start: number;
  end: number;
  cls: number;
  /** Salience in [0,1]. Always 0 for non-WORD tokens — the head skips them. */
  score: number;
  /**
   * Token index of the first WORD of this token's sentence, or -1. Tokens sharing a value
   * are in the same sentence, so this doubles as a sentence id for grouping output.
   */
  sentence: number;
}

function sigmoid(x: number): number {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

/** out[i] = sum_j w[i*cols+j] * x[j] + b[i] */
function matvec(w: Float32Array, b: Float32Array, x: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows);
  for (let i = 0; i < rows; i++) {
    let acc = b[i]!;
    const base = i * cols;
    for (let j = 0; j < cols; j++) acc += w[base + j]! * x[j]!;
    out[i] = acc;
  }
  return out;
}

// --- Embedding -------------------------------------------------------------

/** Feature-bag embedding: sum the selected rows. No vocabulary, no lookup table of words. */
function embed(feats: TokenFeatures[], emb: Float32Array): Float32Array {
  const n = feats.length;
  const out = new Float32Array(n * D);
  for (let t = 0; t < n; t++) {
    const base = t * D;
    for (const row of feats[t]!.rows) {
      const rbase = row * D;
      for (let d = 0; d < D; d++) out[base + d] = out[base + d]! + emb[rbase + d]!;
    }
  }
  return out;
}

// --- Local encoder ---------------------------------------------------------

/**
 * Five-token window (+/-2) with elementwise per-position weights, plus two pointer
 * tokens giving non-adjacent context: the first word of this sentence and the last
 * word of the previous one.
 */
function localEncode(feats: TokenFeatures[], emb: Float32Array, w: Weights): Float32Array {
  const n = feats.length;
  const out = new Float32Array(n * D);
  const half = (WINDOW - 1) / 2;

  for (let t = 0; t < n; t++) {
    const base = t * D;
    for (let d = 0; d < D; d++) out[base + d] = w.localBias[d]!;

    for (let k = 0; k < WINDOW; k++) {
      const src = t + k - half;
      if (src < 0 || src >= n) continue;
      const wbase = k * D;
      const sbase = src * D;
      for (let d = 0; d < D; d++) {
        out[base + d] = out[base + d]! + w.localWin[wbase + d]! * emb[sbase + d]!;
      }
    }

    const ptrs = [feats[t]!.sentenceFirstWord, feats[t]!.prevSentenceLastWord];
    for (let p = 0; p < PTRS; p++) {
      const src = ptrs[p]!;
      if (src < 0) continue;
      const wbase = p * D;
      const sbase = src * D;
      for (let d = 0; d < D; d++) {
        out[base + d] = out[base + d]! + w.localPtr[wbase + d]! * emb[sbase + d]!;
      }
    }

    for (let d = 0; d < D; d++) out[base + d] = Math.tanh(out[base + d]!);
  }

  return out;
}

// --- Learned Blelloch scan -------------------------------------------------

/**
 * Pairwise combine for the linear recurrence s_i = z_i * s_{i-1} + w_i.
 * Composing segment A (left) then B (right): z = zA*zB, w = wB + zB*wA*g.
 *
 * `g` is the per-level, per-channel gate, and it multiplies the **carry** term. Putting it
 * on `z` instead makes every down-sweep parameter dead — the final fold reads only the
 * carry, and the carry never depends on the composed `z`, so those gradients are exactly
 * zero. Gating the carry gives both sweeps real influence.
 *
 * Because `g` varies by tree level this is NOT a mathematically associative operator — it
 * defines one specific balanced binary tree. A sequential loop would build a right-leaning
 * tree and produce different numbers, so Python, TypeScript, and WGSL must all walk this
 * exact same tree shape.
 */
function combineInto(
  outZ: Float32Array,
  outW: Float32Array,
  outOff: number,
  az: Float32Array,
  aw: Float32Array,
  aOff: number,
  bz: Float32Array,
  bw: Float32Array,
  bOff: number,
  levelParam: Float32Array,
  pOff: number,
): void {
  for (let d = 0; d < D; d++) {
    const g = sigmoid(levelParam[pOff + d]!);
    const zA = az[aOff + d]!;
    const wA = aw[aOff + d]!;
    const zB = bz[bOff + d]!;
    const wB = bw[bOff + d]!;
    outZ[outOff + d] = zA * zB;
    outW[outOff + d] = wB + zB * wA * g;
  }
}

/**
 * Inclusive prefix scan over `n` leaves via Blelloch up-sweep + down-sweep.
 *
 * Padding to the next power of two uses the identity element (z=1, w=0) at the tail.
 * Padding cannot affect real positions: the exclusive prefix at position i combines only
 * leaves 0..i-1, and every pad sits strictly to the right of every real leaf.
 */
export function blelloch(leafZ: Float32Array, leafW: Float32Array, n: number, up: Float32Array, down: Float32Array): Float32Array {
  const npad = nextPow2(Math.max(1, n));
  const levels = Math.log2(npad);

  const treeZ: Float32Array[] = [];
  const treeW: Float32Array[] = [];

  const z0 = new Float32Array(npad * D).fill(1);
  const w0 = new Float32Array(npad * D);
  z0.set(leafZ.subarray(0, n * D));
  w0.set(leafW.subarray(0, n * D));
  treeZ.push(z0);
  treeW.push(w0);

  for (let l = 0; l < levels; l++) {
    const size = npad >> (l + 1);
    const pz = new Float32Array(size * D);
    const pw = new Float32Array(size * D);
    const cz = treeZ[l]!;
    const cw = treeW[l]!;
    const pOff = Math.min(l, LEVELS - 1) * D;
    for (let i = 0; i < size; i++) {
      combineInto(pz, pw, i * D, cz, cw, 2 * i * D, cz, cw, (2 * i + 1) * D, up, pOff);
    }
    treeZ.push(pz);
    treeW.push(pw);
  }

  // Down-sweep from an identity root produces the exclusive prefix at every leaf.
  let curZ = new Float32Array(D).fill(1);
  let curW = new Float32Array(D);

  for (let l = levels - 1; l >= 0; l--) {
    const size = npad >> l;
    const nz = new Float32Array(size * D);
    const nw = new Float32Array(size * D);
    const cz = treeZ[l]!;
    const cw = treeW[l]!;
    const pOff = Math.min(l, LEVELS - 1) * D;
    for (let i = 0; i < size / 2; i++) {
      // Left child inherits the parent's exclusive prefix unchanged.
      for (let d = 0; d < D; d++) {
        nz[2 * i * D + d] = curZ[i * D + d]!;
        nw[2 * i * D + d] = curW[i * D + d]!;
      }
      // Right child adds the left subtree's up-sweep total.
      combineInto(nz, nw, (2 * i + 1) * D, curZ, curW, i * D, cz, cw, 2 * i * D, down, pOff);
    }
    curZ = nz;
    curW = nw;
  }

  // Exclusive -> inclusive. No level parameter here; this is a plain fold of the leaf.
  const out = new Float32Array(n * D);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < D; d++) {
      out[i * D + d] = w0[i * D + d]! + z0[i * D + d]! * curW[i * D + d]!;
    }
  }
  return out;
}

/** Minimal-GRU gate: state is (z, (1-z)*tanh(c)), whose pairwise combine composes. */
function gateStates(h: Float32Array, n: number, sw: ScanWeights): { z: Float32Array; w: Float32Array } {
  const z = new Float32Array(n * D);
  const w = new Float32Array(n * D);
  const scratch = new Float32Array(D);

  for (let t = 0; t < n; t++) {
    for (let d = 0; d < D; d++) scratch[d] = h[t * D + d]!;
    const zRaw = matvec(sw.wz, sw.bz, scratch, D, D);
    const cRaw = matvec(sw.wc, sw.bc, scratch, D, D);
    for (let d = 0; d < D; d++) {
      const zi = sigmoid(zRaw[d]!);
      z[t * D + d] = zi;
      w[t * D + d] = (1 - zi) * Math.tanh(cRaw[d]!);
    }
  }
  return { z, w };
}

function reverseSeq(x: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n * D);
  for (let t = 0; t < n; t++) {
    const src = (n - 1 - t) * D;
    for (let d = 0; d < D; d++) out[t * D + d] = x[src + d]!;
  }
  return out;
}

// --- Head ------------------------------------------------------------------

function headScore(local: Float32Array, gf: Float32Array, gb: Float32Array, t: number, w: Weights): number {
  const x = new Float32Array(HEAD_IN);
  for (let d = 0; d < D; d++) {
    x[d] = local[t * D + d]!;
    x[D + d] = gf[t * D + d]!;
    x[2 * D + d] = gb[t * D + d]!;
  }

  const gateRaw = matvec(w.headGateW, w.headGateB, x, GATE, HEAD_IN);
  const hidIn = new Float32Array(HEAD_IN + GATE);
  hidIn.set(x, 0);
  for (let i = 0; i < GATE; i++) hidIn[HEAD_IN + i] = sigmoid(gateRaw[i]!);

  const hidRaw = matvec(w.headHidW, w.headHidB, hidIn, HID, HEAD_IN + GATE);
  const hid = new Float32Array(HID);
  for (let i = 0; i < HID; i++) hid[i] = Math.tanh(hidRaw[i]!);

  let acc = w.headOutB[0]!;
  for (let i = 0; i < HID; i++) acc += w.headOutW[i]! * hid[i]!;
  return sigmoid(acc);
}

// --- Public API ------------------------------------------------------------

export interface Intermediates {
  feats: TokenFeatures[];
  emb: Float32Array;
  local: Float32Array;
  gf: Float32Array;
  gb: Float32Array;
}

/** Run the forward pass and return every intermediate. Used by the golden tests. */
export function forward(text: string, w: Weights): { scored: ScoredToken[]; inter: Intermediates } {
  const feats = extractFeatures(text);
  const n = feats.length;
  const empty = new Float32Array(0);
  if (n === 0) {
    return {
      scored: [],
      inter: { feats, emb: empty, local: empty, gf: empty, gb: empty },
    };
  }

  const emb = embed(feats, w.emb);
  const local = localEncode(feats, emb, w);

  const fwdGate = gateStates(local, n, w.fwd);
  const gf = blelloch(fwdGate.z, fwdGate.w, n, w.fwd.up, w.fwd.down);

  const revLocal = reverseSeq(local, n);
  const bwdGate = gateStates(revLocal, n, w.bwd);
  const gb = reverseSeq(blelloch(bwdGate.z, bwdGate.w, n, w.bwd.up, w.bwd.down), n);

  const scored: ScoredToken[] = feats.map((f, t) => ({
    text: f.text,
    start: f.start,
    end: f.end,
    cls: f.cls,
    score: f.cls === WORD ? headScore(local, gf, gb, t, w) : 0,
    sentence: f.sentenceFirstWord,
  }));

  return { scored, inter: { feats, emb, local, gf, gb } };
}

/** Score every token in `text`. Non-word tokens always score 0. */
export function score(text: string, w: Weights): ScoredToken[] {
  return forward(text, w).scored;
}
