"""Can a reader answer questions about a paragraph from the highlights alone?

Every metric so far measures agreement with an oracle that only agrees with itself 0.704 of
the time. Spearman 0.661 against that is an increasingly weak proxy for the thing we
actually care about, which is whether the highlighted words still carry the meaning.

So measure that directly:

    1. From the FULL paragraph, generate 3 factual questions with short reference answers.
    2. Build several "residues" — the top 20% of words under different methods, in order.
    3. Ask the model to answer the questions from each residue alone, allowing "unknown".
    4. Grade each answer against the reference.

The full paragraph is included as a condition, which is important: it is the sanity check.
If answering from the complete text does not score near 100%, the questions are bad and
nothing else in the table means anything.

Run:  secretspec run --reason "meaning preservation eval" -- \
          python3 -m salience.eval.meaning --limit 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from salience.eval.baselines import build_idf
from salience.features import WORD, ascii_lower, extract_features
from salience.oracle.run import Cache, api_key, call_with_retry
from salience.train.data import build_example, collate
from salience.train.model import SalienceModel

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labeled" / "labels10k.jsonl"
CKPT = ROOT / "data" / "checkpoints" / "model.pt"
CACHE = ROOT / "data" / "labeled" / "meaning_cache.jsonl"
MODEL = "mistral-small-latest"
RATE = 0.20
# Part of the cache key: the system prompts are not, so bump this when they change.
PROMPT_VERSION = "m2"

QUESTION_SYSTEM = """\
You write comprehension questions about a paragraph.

Produce exactly 3 questions. Each one must:
- be answerable by a specific NAMED THING from the paragraph: a name, a place, a number,
  a date, a quantity, or a concrete action. Never a vague summary.
- be self-contained, so someone who has not seen the paragraph understands what is being
  asked. Do not write "what is the issue?" or "what is the subject?"
- have a reference answer that is a specific phrase, not a cryptic fragment
- cover three different facts

If the paragraph is too vague or context-dependent to support three such questions, reply
with an empty array [] instead of inventing weak ones.

Reply with nothing but a JSON array, each item {"q": "...", "a": "..."}."""

ANSWER_SYSTEM = """\
You are shown some text. It may be a complete paragraph, or it may be only a subset of its
words in their original order — in which case it will read like a telegram.

Answer each question using only what the text in front of you supports. Answer in at most
6 words. If it genuinely does not contain the answer, reply exactly: UNKNOWN

Do not guess from general knowledge. But do read fairly: a telegraphic list of words still
states what those words say.

Reply with nothing but a JSON array of 3 strings, one answer per question, in order."""

GRADE_SYSTEM = """\
You grade short answers against a reference answer.

Mark an answer correct if it conveys the same fact as the reference, allowing for
different wording, abbreviation, or extra detail. Mark it incorrect if it states a
different fact, contradicts the reference, or is UNKNOWN.

Reply with nothing but a JSON array of 3 booleans, in order."""


def parse_json(raw: str, expect: int):
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array: {raw[:120]!r}")
    value = json.loads(match.group(0))
    if not isinstance(value, list) or len(value) != expect:
        raise ValueError(f"expected {expect} items, got {value!r}")
    return value


# --- residues ---------------------------------------------------------------


def words_of(text: str) -> list[str]:
    return [f.text for f in extract_features(text) if f.cls == WORD]


def top_k(words: list[str], scores: np.ndarray, rate: float) -> str:
    k = max(1, round(rate * len(words)))
    keep = sorted(np.argsort(-scores)[:k])
    return " ".join(words[i] for i in keep)


@torch.no_grad()
def model_scores(model: SalienceModel, record: dict) -> np.ndarray:
    ex = build_example(record)
    batch = collate([ex])
    probs = torch.sigmoid(model(batch["rows"], batch["ptrs"], batch["lengths"]))[0]
    return probs[batch["is_word"][0]].numpy()


def build_residues(record: dict, model: SalienceModel, idf: dict[str, float], rng: random.Random):
    words = words_of(record["text"])
    n = len(words)
    counts: dict[str, int] = {}
    for w in words:
        counts[ascii_lower(w)] = counts.get(ascii_lower(w), 0) + 1
    default_idf = max(idf.values())

    return {
        "full paragraph": record["text"],
        "model 20%": top_k(words, model_scores(model, record), RATE),
        "oracle 20%": top_k(words, np.array(record["targets"][:n], dtype=float), RATE),
        "tf-idf 20%": top_k(
            words,
            np.array([counts[ascii_lower(w)] * idf.get(ascii_lower(w), default_idf) for w in words]),
            RATE,
        ),
        "random 20%": top_k(words, np.array([rng.random() for _ in words]), RATE),
    }


# --- pipeline ---------------------------------------------------------------


def cached(cache: Cache, key_parts: str, produce) -> str:
    key = hashlib.sha256(key_parts.encode("utf-8")).hexdigest()[:20]
    hit = cache.get(key)
    if hit is not None:
        return hit
    value = produce()
    cache.put(key, value)
    return value


def evaluate_paragraph(key: str, cache: Cache, record: dict, residues: dict) -> dict | None:
    def ask(system: str, user: str, tag: str) -> str:
        return cached(
            cache,
            f"{MODEL}|{PROMPT_VERSION}|{tag}|{record['id']}|{user[:400]}",
            lambda: call_with_retry_system(key, system, user),
        )

    try:
        raw = ask(QUESTION_SYSTEM, record["text"], "q")
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        qa = json.loads(match.group(0)) if match else []
        if len(qa) != 3:
            return "declined"  # too vague to question fairly
        questions = [item["q"] for item in qa]
        references = [item["a"] for item in qa]
    except (ValueError, KeyError, TypeError, AttributeError):
        return None

    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    out: dict[str, int] = {}
    for name, residue in residues.items():
        user = f"Text:\n{residue}\n\nQuestions:\n{numbered}"
        try:
            answers = parse_json(ask(ANSWER_SYSTEM, user, f"a|{name}"), 3)
            grade_user = "\n\n".join(
                f"Q: {q}\nReference: {r}\nAnswer: {a}"
                for q, r, a in zip(questions, references, answers)
            )
            verdicts = parse_json(ask(GRADE_SYSTEM, grade_user, f"g|{name}"), 3)
            out[name] = sum(1 for v in verdicts if v is True)
        except (ValueError, KeyError, TypeError):
            return None
    return out


def call_with_retry_system(key: str, system: str, user: str) -> str:
    """oracle.run.call_with_retry hardcodes the salience SYSTEM prompt; this varies it."""
    import salience.oracle.run as runner  # noqa: PLC0415

    original = runner.SYSTEM
    try:
        runner.SYSTEM = system
        return call_with_retry(key, MODEL, user)
    finally:
        runner.SYSTEM = original


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from salience.train.data import load_examples, split  # noqa: PLC0415

    with args.labels.open(encoding="utf-8") as fh:
        records = {json.loads(line)["id"]: json.loads(line) for line in fh if line.strip()}
    _, val = split(load_examples(args.labels), 0.15, args.seed)
    # split() groups by domain, so val is ordered email, essays, news, technical, wikipedia.
    # Taking a prefix would sample one register; shuffle first.
    random.Random(args.seed).shuffle(val)
    chosen = [records[ex.pid] for ex in val[: args.limit]]

    model = SalienceModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()

    print("computing idf...", end=" ", flush=True)
    idf = build_idf()
    print(f"{len(idf):,} terms")

    rng = random.Random(args.seed)
    residues = {r["id"]: build_residues(r, model, idf, rng) for r in chosen}
    conditions = list(next(iter(residues.values())).keys())

    key = api_key()
    cache = Cache(CACHE)
    print(f"{len(chosen)} paragraphs x {len(conditions)} conditions x 3 questions")
    print(f"cache holds {len(cache):,} responses\n")

    results: list[dict] = []
    done = failed = declined = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_paragraph, key, cache, r, residues[r["id"]]): r for r in chosen
        }
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result == "declined":
                declined += 1
            elif result is None:
                failed += 1
            else:
                results.append(result)
            if done % 10 == 0:
                rate = done / max(1e-9, time.time() - start)
                print(f"  {done}/{len(chosen)}  eta {(len(chosen)-done)/max(rate,1e-9)/60:.1f}m", end="\r")
    cache.close()

    # A question only measures the highlights if it is answerable from the full text.
    # Where the complete paragraph scores less than 3/3 the question is at fault, not the
    # highlighting, so those paragraphs are excluded rather than allowed to add noise.
    valid = [r for r in results if r["full paragraph"] == 3]

    print(f"\n\n{len(chosen)} sampled · {declined} too vague to question · {failed} malformed · "
          f"{len(results)} scored\n{len(valid)} of those answerable from the full paragraph")
    print(f"question validity: {len(valid) / max(1, len(results)):.0%}\n")

    print(f"{'condition':<18}{'correct':>10}{'of':>7}{'accuracy':>11}")
    print("-" * 46)
    for c in conditions:
        got = sum(r[c] for r in valid)
        total = 3 * len(valid)
        bar = "#" * round(30 * got / max(1, total))
        print(f"{c:<18}{got:>10}{total:>7}{got / max(1, total):>10.1%}  {bar}")

    if valid:
        model_acc = sum(r["model 20%"] for r in valid) / (3 * len(valid))
        rand_acc = sum(r["random 20%"] for r in valid) / (3 * len(valid))
        print(f"\nReading 20% of the words answers {model_acc:.0%} of questions that the full")
        print(f"paragraph answers, against {rand_acc:.0%} for a random 20%.")


if __name__ == "__main__":
    main()
