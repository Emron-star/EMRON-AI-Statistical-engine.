/* EMRON AI Butler proxy (Vercel Function, standard Web handler).
   Holds the NVIDIA API key as a server-side environment variable so it never reaches the browser, and forwards a validated chat request to
   NVIDIA's OpenAI-compatible endpoint. The key is read from process.env.NVIDIA_API_KEY only.

   Environment variables (Vercel > your project > Settings > Environment Variables; use the Secret type for the first two):
     NVIDIA_API_KEY        required
     BUTLER_ACCESS_CODE    strongly advised: a passphrase the page must send, so strangers cannot spend your NVIDIA credits
     NVIDIA_MODEL          optional, default meta/llama-3.3-70b-instruct
     ALLOWED_ORIGINS       optional, comma separated, e.g. https://www.mysite.com (calls from the site's own address are always allowed)
     UPSTREAM_TIMEOUT_MS   optional, default 25000
     RATE_LIMIT_PER_10MIN  optional, default 40 per client address (best effort; instances do not share memory)
   After changing any variable, redeploy: changes only apply to new deployments. */

import crypto from 'node:crypto';

const NVIDIA_URL = 'https://integrate.api.nvidia.com/v1/chat/completions';
const DEFAULT_MODEL = 'meta/llama-3.3-70b-instruct';
const MAX_BODY_BYTES = 60000;
const MAX_MESSAGES = 20;
const MAX_MESSAGE_CHARS = 16000;
const MAX_TOTAL_CHARS = 32000;
const hits = new Map();

const reply = (status, obj, extra) =>
  new Response(status === 204 ? null : JSON.stringify(obj), {
    status,
    headers: Object.assign({ 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' }, extra || {}),
  });

const same = (a, b) => {
  const x = Buffer.from(String(a)), y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
};

function originVerdict(request) {
  const origin = request.headers.get('origin');
  if (!origin) return { ok: true, origin: null }; // non-browser caller: the access code is the guard
  let oh = '';
  try { oh = new URL(origin).host; } catch (_) { return { ok: false, origin }; }
  const hosts = [request.headers.get('host')];
  try { hosts.push(new URL(request.url).host); } catch (_) { /* ignore */ }
  if (oh && hosts.includes(oh)) return { ok: true, origin };
  const list = String(process.env.ALLOWED_ORIGINS || '').split(',').map((s) => s.trim().replace(/\/$/, '')).filter(Boolean);
  return { ok: list.includes(origin.replace(/\/$/, '')), origin };
}

function rateLimited(ip) {
  const limit = Math.max(5, Number(process.env.RATE_LIMIT_PER_10MIN) || 40);
  const now = Date.now(), win = 10 * 60 * 1000;
  const arr = (hits.get(ip) || []).filter((t) => now - t < win);
  arr.push(now);
  hits.set(ip, arr);
  if (hits.size > 500) for (const [k, v] of hits) if (!v.length || now - v[v.length - 1] > win) hits.delete(k);
  return arr.length > limit;
}

function validate(body) {
  let b;
  try { b = JSON.parse(body); } catch (_) { return { error: 'The request body is not valid JSON.' }; }
  if (!b || !Array.isArray(b.messages) || !b.messages.length || b.messages.length > MAX_MESSAGES) return { error: 'messages must be a list of 1 to ' + MAX_MESSAGES + ' items.' };
  let total = 0;
  const messages = [];
  for (const m of b.messages) {
    if (!m || !['system', 'user', 'assistant'].includes(m.role) || typeof m.content !== 'string') return { error: 'Each message needs a role (system, user or assistant) and text content.' };
    if (m.content.length > MAX_MESSAGE_CHARS) return { error: 'A message is too long.' };
    total += m.content.length;
    messages.push({ role: m.role, content: m.content });
  }
  if (total > MAX_TOTAL_CHARS) return { error: 'The conversation is too long.' };
  const model = typeof b.model === 'string' && /^[\w.\-]+\/[\w.\-]+$/.test(b.model) ? b.model : (process.env.NVIDIA_MODEL || DEFAULT_MODEL);
  const t = Number(b.temperature), mt = Number(b.max_tokens);
  return {
    payload: {
      model, messages, stream: false,
      temperature: Number.isFinite(t) ? Math.min(1.2, Math.max(0, t)) : 0.3,
      max_tokens: Number.isFinite(mt) ? Math.min(1200, Math.max(16, Math.round(mt))) : 600,
    },
  };
}

async function handle(request) {
  const ov = originVerdict(request);
  const cors = ov.origin && ov.ok
    ? { 'Access-Control-Allow-Origin': ov.origin, Vary: 'Origin', 'Access-Control-Allow-Headers': 'content-type, x-butler-code', 'Access-Control-Allow-Methods': 'POST, OPTIONS' }
    : { Vary: 'Origin' };
  if (request.method === 'OPTIONS') return ov.ok ? reply(204, null, cors) : reply(403, { error: 'Origin not allowed.' });
  if (request.method !== 'POST') return reply(405, { error: 'Use POST.' }, cors);
  if (!ov.ok) return reply(403, { error: 'Origin not allowed. Add it to ALLOWED_ORIGINS in the proxy settings.' }, cors);

  const code = process.env.BUTLER_ACCESS_CODE;
  if (code && !same(request.headers.get('x-butler-code') || '', code)) return reply(401, { error: 'Access code missing or wrong.' }, cors);

  const key = process.env.NVIDIA_API_KEY;
  if (!key) return reply(500, { error: 'NVIDIA_API_KEY is not set in the proxy settings.' }, cors);

  const ip = String(request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown').split(',')[0].trim();
  if (rateLimited(ip)) return reply(429, { error: 'Too many requests. Please wait a few minutes.' }, cors);

  if (Number(request.headers.get('content-length')) > MAX_BODY_BYTES) return reply(413, { error: 'The request is too large.' }, cors);
  let raw = '';
  try { raw = await request.text(); } catch (_) { return reply(400, { error: 'Could not read the request body.' }, cors); }
  if (Buffer.byteLength(raw) > MAX_BODY_BYTES) return reply(413, { error: 'The request is too large.' }, cors);
  const v = validate(raw);
  if (v.error) return reply(400, { error: v.error }, cors);

  const ctl = new AbortController();
  const to = setTimeout(() => ctl.abort(), Math.min(60000, Math.max(2000, Number(process.env.UPSTREAM_TIMEOUT_MS) || 25000)));
  try {
    const res = await fetch(NVIDIA_URL, {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + key, 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(v.payload),
      signal: ctl.signal,
    });
    let data = null;
    try { data = await res.json(); } catch (_) { /* upstream error page */ }
    if (!res.ok) {
      const d = data && (data.detail || data.title || (data.error && (data.error.message || data.error)));
      const detail = d ? String(d).slice(0, 200) : '';
      if (res.status === 401 || res.status === 403) return reply(502, { error: 'NVIDIA rejected the key or this model (HTTP ' + res.status + '). Check NVIDIA_API_KEY and that your account can use ' + v.payload.model + '.' }, cors);
      if (res.status === 404) return reply(502, { error: 'NVIDIA does not know the model ' + v.payload.model + '. Pick another in the butler settings.' }, cors);
      if (res.status === 429) return reply(429, { error: 'NVIDIA is rate limiting this key. Please wait.' }, cors);
      return reply(502, { error: 'NVIDIA returned HTTP ' + res.status + (detail ? ': ' + detail : '') + '.' }, cors);
    }
    const choice = data && data.choices && data.choices[0];
    const content = choice && choice.message && typeof choice.message.content === 'string' ? choice.message.content : '';
    if (!content.trim()) return reply(502, { error: 'The model returned an empty reply.' }, cors);
    return reply(200, { content, model: v.payload.model, usage: data.usage || null }, cors);
  } catch (e) {
    if (e && e.name === 'AbortError') return reply(504, { error: 'NVIDIA took too long to answer. Choose a faster model in the butler settings, or raise UPSTREAM_TIMEOUT_MS.' }, cors);
    return reply(502, { error: 'Could not reach NVIDIA from the proxy.' }, cors);
  } finally {
    clearTimeout(to);
  }
}

export default { fetch: handle };
