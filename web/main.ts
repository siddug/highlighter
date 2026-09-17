import weightFile from '../weights/model.json';
import { score, type ScoredToken } from '../ts/src/cpu';
import { WORD } from '../ts/src/features';
import { PARAM_COUNT } from '../ts/src/model';
import { decodeWeights, type WeightFile } from '../ts/src/weights';
import { SAMPLES } from './samples';

type Runtime = 'browser' | 'server';

const weights = decodeWeights(weightFile as WeightFile);

const el = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
};

const input = el<HTMLTextAreaElement>('input');
const rate = el<HTMLInputElement>('rate');
const rateOut = el<HTMLOutputElement>('rateOut');
const output = el('output');
const digest = el('digest');
const hover = el('hover');
const notice = el('notice');
const sampleRow = el('sampleRow');

const stat = {
  tokens: el('statTokens'),
  words: el('statWords'),
  picked: el('statPicked'),
  ms: el('statMs'),
  delta: el('statDelta'),
  deltaWrap: el('statDeltaWrap'),
};

const buttons: Record<Runtime, HTMLButtonElement> = {
  browser: el<HTMLButtonElement>('runBrowser'),
  server: el<HTMLButtonElement>('runServer'),
};

el('metaParams').textContent = PARAM_COUNT.toLocaleString();

let runtime: Runtime = 'browser';
let activeSample = 0;
/** Guards against a slow response overwriting a newer one. */
let requestSeq = 0;

// --- scoring ---------------------------------------------------------------

interface Scored {
  tokens: ScoredToken[];
  ms: number;
  /** Largest per-word disagreement with the browser, when the server ran it. */
  delta?: number;
}

function scoreInBrowser(text: string): Scored {
  const t0 = performance.now();
  const tokens = score(text, weights);
  return { tokens, ms: performance.now() - t0 };
}

async function scoreOnServer(text: string): Promise<Scored> {
  const response = await fetch('/api/score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(detail.error ?? `HTTP ${response.status}`);
  }
  const payload = (await response.json()) as { tokens: ScoredToken[]; ms: number };

  // The conformance tests claim PyTorch and the browser agree to within 2e-4. Show it.
  const local = scoreInBrowser(text).tokens;
  let delta = 0;
  for (let i = 0; i < Math.min(local.length, payload.tokens.length); i++) {
    delta = Math.max(delta, Math.abs(local[i]!.score - payload.tokens[i]!.score));
  }

  return { tokens: payload.tokens, ms: payload.ms, delta };
}

/** Take the top `fraction` of words by score — the operating point we evaluate at (P@k). */
function pickTop(tokens: ScoredToken[], fraction: number): Set<number> {
  const words = tokens
    .map((t, i) => ({ i, score: t.score }))
    .filter((_, i) => tokens[i]!.cls === WORD);
  if (words.length === 0) return new Set();
  const k = Math.max(1, Math.round(words.length * fraction));
  words.sort((a, b) => b.score - a.score);
  return new Set(words.slice(0, k).map((w) => w.i));
}

// --- rendering -------------------------------------------------------------

/** How long the marks take to sweep across the paragraph, regardless of its length. */
const SWEEP_MS = 220;

function paint(result: Scored, sweep = false): void {
  const { tokens, ms, delta } = result;
  const fraction = Number(rate.value) / 100;
  const picked = pickTop(tokens, fraction);

  output.classList.toggle('is-sweeping', sweep);
  output.replaceChildren();
  let marked = 0;
  for (const [i, token] of tokens.entries()) {
    if (token.cls !== WORD) {
      output.append(document.createTextNode(token.text));
      continue;
    }
    const span = document.createElement('span');
    const on = picked.has(i);
    span.className = `w ${on ? 'on' : 'off'}`;
    if (on) {
      span.style.setProperty('--a', (0.16 + 0.55 * token.score).toFixed(3));
      // Spread the delay across the whole sweep rather than a fixed per-item step, so
      // long paragraphs do not turn a flourish into a wait.
      if (sweep && picked.size > 1) {
        span.style.setProperty('--d', `${Math.round((marked / (picked.size - 1)) * SWEEP_MS)}ms`);
      }
      marked++;
    }
    span.textContent = token.text;
    span.dataset.score = token.score.toFixed(4);
    output.append(span);
  }

  // Break the residue at sentence boundaries. Without it the words run together and you
  // lose which facts belong to each other — "failed twice lock wasn't" reads as one claim
  // when it is two.
  digest.replaceChildren();
  const fullStop = () => {
    const stop = document.createElement('span');
    stop.className = 'stop';
    stop.textContent = '.';
    return stop;
  };

  let sentence = -2;
  for (const i of [...picked].sort((a, b) => a - b)) {
    const token = tokens[i]!;
    if (sentence !== -2) {
      // Separator goes before the word, so the stop sits tight against the previous one
      // rather than floating after a trailing space.
      if (token.sentence !== sentence) digest.append(fullStop());
      digest.append(' ');
    }
    const word = document.createElement('b');
    word.textContent = token.text;
    digest.append(word);
    sentence = token.sentence;
  }
  if (picked.size > 0) digest.append(fullStop());

  const wordCount = tokens.filter((t) => t.cls === WORD).length;
  stat.tokens.textContent = String(tokens.length);
  stat.words.textContent = String(wordCount);
  stat.picked.textContent = `${picked.size} · ${wordCount ? Math.round((100 * picked.size) / wordCount) : 0}%`;
  stat.ms.textContent = `${ms.toFixed(1)} ms`;

  stat.deltaWrap.hidden = delta === undefined;
  if (delta !== undefined) stat.delta.textContent = delta.toExponential(1);
}

function setNotice(message: string | null): void {
  notice.hidden = message === null;
  if (message) notice.textContent = message;
}

let lastResult: Scored | null = null;

/**
 * `sweep` is a frequency judgement, not a taste one. Picking a sample or loading the page
 * happens once, so the marks can animate on. Typing and dragging the rate slider fire tens
 * of times a second, and an animation there would charge its cost on every keystroke.
 */
async function render(sweep = false): Promise<void> {
  rateOut.textContent = `${rate.value}%`;
  const text = input.value;

  if (text.trim() === '') {
    lastResult = null;
    paint({ tokens: [], ms: 0 });
    return;
  }

  const seq = ++requestSeq;

  if (runtime === 'server') {
    try {
      const result = await scoreOnServer(text);
      if (seq !== requestSeq) return;
      setNotice(null);
      lastResult = result;
      paint(result, sweep);
      return;
    } catch (error) {
      if (seq !== requestSeq) return;
      setNotice(
        `Server unavailable (${(error as Error).message}) — falling back to the browser. ` +
          `Start it with: python3 serve.py`,
      );
      selectRuntime('browser');
    }
  }

  const result = scoreInBrowser(text);
  if (seq !== requestSeq) return;
  lastResult = result;
  paint(result, sweep);
}

/** Repaint from cached scores. Changing the rate does not need a re-run. */
function repaint(): void {
  rateOut.textContent = `${rate.value}%`;
  if (lastResult) paint(lastResult);
}

// --- wiring ----------------------------------------------------------------

function selectRuntime(next: Runtime): void {
  runtime = next;
  for (const [name, button] of Object.entries(buttons)) {
    button.classList.toggle('is-active', name === next);
  }
  el('metaRuntime').textContent = next;
  if (next === 'browser') stat.deltaWrap.hidden = true;
}

function selectSample(index: number): void {
  activeSample = index;
  for (const [i, chip] of [...sampleRow.children].entries()) {
    chip.classList.toggle('is-active', i === index);
  }
}

for (const [i, sample] of SAMPLES.entries()) {
  const chip = document.createElement('button');
  chip.type = 'button';
  chip.className = 'chip';
  chip.textContent = sample.label;
  chip.addEventListener('click', () => {
    selectSample(i);
    input.value = sample.text;
    void render(true);
  });
  sampleRow.append(chip);
}

let debounce: number | undefined;
input.addEventListener('input', () => {
  selectSample(-1);
  window.clearTimeout(debounce);
  // Browser inference is sub-20ms, but the server round-trip should not fire per keystroke.
  debounce = window.setTimeout(() => void render(), runtime === 'server' ? 220 : 0);
});

rate.addEventListener('input', repaint);

buttons.browser.addEventListener('click', () => {
  selectRuntime('browser');
  setNotice(null);
  void render();
});
buttons.server.addEventListener('click', () => {
  selectRuntime('server');
  void render();
});

el('clearBtn').addEventListener('click', () => {
  input.value = '';
  selectSample(-1);
  void render();
  input.focus();
});

output.addEventListener('mouseover', (event) => {
  const target = event.target as HTMLElement;
  if (!target.classList?.contains('w')) return;
  target.classList.add('peek');
  hover.textContent = `${target.textContent} · ${target.dataset.score}`;
});

output.addEventListener('mouseout', (event) => {
  (event.target as HTMLElement).classList?.remove('peek');
  hover.textContent = 'hover a word for its score';
});

el('copyBtn').addEventListener('click', async () => {
  const button = el<HTMLButtonElement>('copyBtn');
  const text = digest.textContent ?? '';
  if (text === '') return;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    return;
  }
  // Motion is never the only feedback channel — the label carries the state on its own.
  button.textContent = 'Copied';
  window.setTimeout(() => {
    button.textContent = 'Copy';
  }, 1400);
});

/**
 * The PyTorch server only exists when someone runs `python3 serve.py` next to this page.
 * On the hosted build there is no `/api`, so the runtime toggle stays hidden rather than
 * offering a choice that resolves to an error and a fallback notice.
 */
async function detectServer(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 1500);
    const response = await fetch('/api/health', { signal: controller.signal });
    window.clearTimeout(timer);
    if (!response.ok) return false;
    // A 200 is not enough: static hosts commonly answer unknown paths with the index
    // page, which would reveal a toggle backed by nothing. Require the real payload.
    const body = (await response.json()) as { ok?: boolean };
    return body.ok === true;
  } catch {
    return false;
  }
}

selectRuntime('browser');
selectSample(activeSample);
input.value = SAMPLES[activeSample]!.text;
void render(true);

void detectServer().then((available) => {
  if (!available) return;
  el('runtimeControl').hidden = false;
  // ?runtime=server is handy for sharing a link that demonstrates both runtimes agreeing.
  if (new URLSearchParams(location.search).get('runtime') === 'server') {
    selectRuntime('server');
    void render();
  }
});
