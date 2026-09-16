# Feature Specification v2

This document is **normative**. `ts/src/features.ts` and `salience/features.py` are two
implementations of this one spec, and `shared/golden/*.jsonl` proves they agree.

When the two implementations disagree, this document decides who is wrong. When this
document is ambiguous, that is a bug in this document — fix it here first, then fix the code.

Design inherited from Vercel's `gpu-lexer`: there is **no learned vocabulary**. Every token is
reduced to a handful of small integers. Each integer indexes a row of one shared embedding
table, and the rows are **summed**. Feature engineering does the work a tokenizer vocabulary
would normally do.

---

## 1. Determinism rules

These exist because the two implementations run in different languages, and a silent
divergence here poisons everything downstream.

1. **All arithmetic on hashes is unsigned 32-bit.**
   - TypeScript: use `Math.imul` for every multiply and `>>> 0` after *every* operation.
   - Python: mask with `& 0xFFFFFFFF` after every operation.
2. **Lowercasing is ASCII-only.** Map `A`–`Z` (U+0041–U+005A) to `a`–`z`. Leave every other
   code point untouched. Do **not** use `String.prototype.toLowerCase()` or `str.lower()` —
   they disagree on Turkish dotted İ, German ß, and others.
3. **Hash input is the UTF-8 byte sequence** of the ASCII-lowercased surface form, hashed byte
   by byte — not code point by code point.
4. **Character class membership uses Unicode general categories**, not language built-ins:
   - *letter* = category starts with `L` or `M`
   - *digit* = category starts with `N`
   - TypeScript: `/\p{L}|\p{M}/u` and `/\p{N}/u`. Python: `unicodedata.category()`.
5. **Newlines are normalized before tokenizing**: `\r\n` → `\n`, lone `\r` → `\n`.
6. Indices, buckets, and row offsets are integers. Never compare them as floats.

---

## 2. Tokenization

One linear left-to-right pass. Every character of the input belongs to exactly one token, so
concatenating all token texts in order reproduces the normalized input exactly. This is a hard
invariant — the golden tests assert it.

### Token classes

| Class | Name | Rule |
|---|---|---|
| 0 | `WORD` | A run of letters/digits, which may contain internal connectors (see below). |
| 1 | `SPACE` | A maximal run of spaces (U+0020) and tabs (U+0009). |
| 2 | `NEWLINE` | Exactly one `\n`. Consecutive newlines emit separate tokens. |
| 3 | `PUNCT` | Any other single character, emitted one token at a time. |

**Internal connectors.** A character inside a `WORD` run is absorbed rather than ending it when:

| Connector | Absorbed when |
|---|---|
| `'` (U+0027), `-` (U+002D) | the preceding and following characters are both letters or digits |
| `.` (U+002E), `,` (U+002C) | the preceding and following characters are both **digits** |

The digit-only rule for `.` and `,` exists because numbers are facts, and facts are exactly
what this model is supposed to highlight. Splitting `1.6` into `1`, `.`, `6` produced
highlight residue like "grown 1 6 kilometers year 1 2 kilometers" during oracle piloting —
the quantity is destroyed. Restricting the rule to digits keeps `3.14`, `10,000`, and `1.6`
intact while leaving sentence-ending periods, `e.g.`, and `U.S.` alone, since those have a
letter on at least one side.

Notes:

- `don't` and `state-of-the-art` are single `WORD` tokens. A trailing `-` in `well-` is not
  internal, so it becomes a separate `PUNCT`.
- Only U+002D counts as a hyphen. En dash, em dash, and minus sign are `PUNCT`.
- `Section 1. The` is unaffected: the `.` has a space after it, not a digit.
- A `NEWLINE` per line break (rather than a run) means a blank line surfaces as two adjacent
  `NEWLINE` tokens, which the paragraph-break flag keys off.
- `PUNCT` is one character per token, so `?!` is two tokens. Simpler to specify, and the local
  window sees both anyway.

### Sentence segmentation

Deliberately dumb, and deterministic. It is a *feature*, not ground truth — abbreviations like
`Dr.` will split incorrectly and that is acceptable.

A sentence boundary occurs after a `PUNCT` token whose character is `.`, `!`, `?`, or `…`,
skipping over any immediately following closing punctuation (`"`, `'`, `)`, `]`, `}`, `»`, `”`,
`’`) and any `SPACE`, when either:

- the next token is a `WORD` starting with an uppercase letter (category `Lu`) or a digit, or
- the input ends.

A `NEWLINE` token also ends the current sentence unconditionally.

---

## 3. Feature fields

Each token produces the integer fields below. Fields marked **word-only** are computed for
`WORD` tokens and are **omitted** (contribute no embedding row) for other classes.

| # | Field | Bits | Values | Word-only |
|---|---|---|---|---|
| 1 | token class | 3 | 0–3 (4 used, 8 reserved) | no |
| 2 | hash A | 8 | 0–255 | yes |
| 3 | hash B | 8 | 0–255 | yes |
| 4 | suffix hash | 6 | 0–63 | yes |
| 5 | prefix hash | 5 | 0–31 | yes |
| 6 | stopword bucket | 4 | 0–15 | yes |
| 7 | flags | — | 9 independent bits | no |
| 8 | length bucket | 4 | 0–15 | yes |
| 9 | sentence-position bucket | 5 | 0–31 | no |
| 10 | document-position bucket | 5 | 0–31 | no |
| 11 | within-document frequency bucket | 4 | 0–15 | yes |

### 3.1 Hashes

```
FNV(bytes, basis, prime):
    h = basis
    for b in bytes:
        h = ((h XOR b) * prime) mod 2^32
    return h

MIX(h):                                  # xorshift-multiply avalanche
    h = h XOR (h >>> 16)
    h = (h * 2246822507) mod 2^32
    h = h XOR (h >>> 13)
    h = (h * 3266489909) mod 2^32
    h = h XOR (h >>> 16)
    return h
```

Let `s` = UTF-8 bytes of the ASCII-lowercased token text.

- `hashA = MIX(FNV(s, 2166136261, 16777619)) & 0xFF`
- `hashB = MIX(FNV(s, 1166136321, 2246822519)) & 0xFF`
- `suffixHash = MIX(FNV(last 3 bytes of s, 2166136261, 16777619)) & 0x3F`
- `prefixHash = MIX(FNV(first 3 bytes of s, 2166136261, 16777619)) & 0x1F`

`1166136321` is `2166136261 - 999999940`, chosen only to be a distinct basis; there is nothing
magic about it. Prefix and suffix slice the **byte** sequence, not code points, and use the
whole sequence when it is shorter than 3 bytes.

Two independent 8-bit hashes give 65,536 effective buckets through the sum of two 256-row
embeddings, at a cost of 512 rows instead of 65,536. Collisions are real and intentional — the
local window and the scan disambiguate.

Suffix is an English part-of-speech proxy (`-ing`, `-ion`, `-ly`, `-ed`), which is why it gets
its own field rather than relying on hash A and B to encode it implicitly.

### 3.2 Stopword bucket

`shared/spec/stopwords.txt` holds a fixed, ordered list of common English words, most frequent
first. Both implementations load this exact file.

- Not in the list → `0`
- In the list at zero-based rank `r` → `min(15, 1 + floor(r / 16))`

Stopwords are almost never "important", so this field lets the model learn a strong prior
cheaply — but as a graded bucket rather than a single bit, because `not` and `but` carry
meaning while `the` does not.

### 3.3 Flags

Nine independent bits. Each set bit contributes its own embedding row; clear bits contribute
nothing.

| Bit | Name | Meaning |
|---|---|---|
| 0 | `CAPITALIZED` | First character is category `Lu`, and the token is not all-caps. |
| 1 | `ALL_CAPS` | Two or more letters and every letter is `Lu`. |
| 2 | `HAS_DIGIT` | Contains at least one `N` character. |
| 3 | `ALL_DIGITS` | Every character is `N`. |
| 4 | `HAS_HYPHEN` | Contains U+002D. |
| 5 | `HAS_APOSTROPHE` | Contains U+0027. |
| 6 | `SENTENCE_START` | First `WORD` token of its sentence. |
| 7 | `PARAGRAPH_START` | First `WORD` token after two or more consecutive `NEWLINE`s, or at input start. |
| 8 | `BEFORE_SENTENCE_END` | The next non-`SPACE` token is sentence-final punctuation. |

`CAPITALIZED` and `ALL_CAPS` are mutually exclusive by construction so the model can tell
`Apple` from `NASA`.

### 3.4 Buckets

**Length** — in code points, not bytes:

```
1,2,3,4,5,6,7,8 → 0..7
9-10 → 8    11-12 → 9    13-15 → 10   16-19 → 11
20-24 → 12  25-31 → 13   32-47 → 14   48+ → 15
```

**Sentence position** — `i` = index of this token among all tokens of its sentence, `n` = total:
`min(31, floor(32 * i / max(1, n)))`

**Document position** — `j` = index among all tokens, `N` = total: `min(31, floor(32 * j / max(1, N)))`

Relative rather than absolute, so a 30-word and a 300-word document are comparable. Early
position matters because topic sentences lead.

**Within-document frequency** — `c` = number of tokens in the document whose ASCII-lowercased
text equals this one's:

```
1,2,3,4,5,6,7,8 → 0..7
9-10 → 8    11-12 → 9    13-16 → 10   17-24 → 11
25-32 → 12  33-48 → 13   49-64 → 14   65+ → 15
```

A poor-man's term frequency, computable at inference time with no corpus statistics. This is
the field most likely to need revision once we see real labels.

---

## 4. Embedding row map

One table of **737 rows × 32 dimensions = 23,584 parameters**. A token's embedding is the sum
of the rows its fields select.

| Rows | Count | Field |
|---|---|---|
| 0–7 | 8 | token class |
| 8–263 | 256 | hash A |
| 264–519 | 256 | hash B |
| 520–583 | 64 | suffix hash |
| 584–615 | 32 | prefix hash |
| 616–631 | 16 | stopword bucket |
| 632–640 | 9 | flags (one row per bit) |
| 641–656 | 16 | length bucket |
| 657–688 | 32 | sentence-position bucket |
| 689–720 | 32 | document-position bucket |
| 721–736 | 16 | within-document frequency bucket |

A `SPACE` token therefore sums exactly three rows (class, sentence position, document
position), while a `WORD` token sums ten plus one per set flag.

Implementations emit, per token, a **sorted list of row indices**. The golden fixtures compare
these lists as integers. Never compare embeddings — compare the indices that produce them.

---

## 5. Model shape (informative)

Not part of the tokenizer contract, recorded here so the parameter budget stays visible.

| Component | Parameters |
|---|---|
| Embedding table (737 × 32) | 23,584 |
| Local encoder: bias 32 + 5-token window 5×32 + 2 pointers 2×32 | 256 |
| Scan gates, ×2 directions: two 32→32 maps + biases | 4,224 |
| Scan level params, ×2 directions: up-sweep 12×32 + down-sweep 12×32 | 1,536 |
| Head: gate 16×96+16, hidden 72×112+72, output 1×72+1 | 9,761 |
| **Total** | **39,361** |

Under the 40k budget, with gpu-lexer's proportions preserved: the embedding is ~60% of
the model.

Unlike gpu-lexer we run the scan **bidirectionally** — a forward pass and a backward pass
with separate parameters, concatenated into the head. A prefix scan alone gives only left
context, which suffices for syntax highlighting but not for prose: whether a word matters
often depends on what the sentence goes on to say.

### Initialization requirement

The scan's forget-gate bias (`fwd.bz`, `bwd.bz`) **must** be initialized to about **+2.0**,
putting `sigmoid(z)` near 0.88 at the start of training.

This is not a tuning nicety. At the default near-zero bias the gate sits at ~0.5 and the
carry decays as `0.5^distance`, underflowing float32 within roughly 100 tokens. Measured on
this implementation with random weights: a word 8 tokens away moved a score by `1.6e-7`, and
one ~90 tokens away moved it by exactly `0`. The model would begin with no long-range context
and no gradient with which to acquire any. `FORGET_BIAS_INIT` in `ts/src/model.ts` is the
shared constant.

The two **pointer tokens** in the local encoder are the prose analogue of gpu-lexer's
non-adjacent context (which pointed at line starts and matched delimiters). Ours point at:

1. the first `WORD` of the current sentence,
2. the last `WORD` of the previous sentence.

The head emits **one logit** (salience) rather than gpu-lexer's nine class logits. As in
gpu-lexer, the head is **skipped entirely** for non-`WORD` tokens — they score 0 and cost nothing.

---

## 6. Versioning

This is **v2**. v1 split decimals such as `1.6` into three tokens, destroying quantities in the highlight residue; v2 added the digit-only internal connectors in section 2. Any change to hashes, row map, bucket edges, or tokenization rules is a
breaking change: bump the version, regenerate `shared/golden/`, and retrain. The version string
is written into `weights/model.json` and checked at load time.
