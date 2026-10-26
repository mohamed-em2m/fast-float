/**
 * FLOAT Studio API Client
 *
 * The backend is reached through the same-origin `/api` reverse proxy (see
 * vite.config.js), so the browser talks to whatever host served the page —
 * localhost, a LAN IP, or an ngrok tunnel. Hard-coding `http://localhost:8000`
 * breaks every remote browser: there, `localhost` is the *user's* machine.
 */

import { fetchJson, fetchWithTimeout, HttpError } from './http';
import { getStored, removeStored, setStored } from './storage';

/** Path prefix proxied to the FastAPI backend by Vite (and any prod proxy). */
const API_PROXY_PATH = '/api';

/** Hosts that mean "this very machine" and are useless to a remote browser. */
const LOOPBACK_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0'];

function isLoopbackHost(hostname) {
  return LOOPBACK_HOSTS.includes(hostname);
}

function pageOrigin() {
  return typeof window !== 'undefined' ? window.location.origin : 'http://localhost';
}

export function isLoopbackUrl(value) {
  try {
    return isLoopbackHost(new URL(value, pageOrigin()).hostname);
  } catch {
    return false;
  }
}

/** True when the browser tab itself is served from a loopback host. */
export function isPageOnLoopback() {
  if (typeof window === 'undefined') return false;
  return isLoopbackHost(window.location.hostname);
}

/** Same-origin API base that the dev/prod proxy forwards to the backend. */
function sameOriginApiBase() {
  if (typeof window === 'undefined') return 'http://localhost:8000';
  return `${window.location.origin}${API_PROXY_PATH}`;
}

/**
 * Default API base. Prefer the same-origin proxy so a remote browser (ngrok,
 * LAN) reaches the real server; a loopback VITE_API_URL is ignored because it
 * would trap the browser on its own machine. A non-loopback VITE_API_URL wins.
 */
export const DEFAULT_API_URL =
  import.meta.env.VITE_API_URL && !isLoopbackUrl(import.meta.env.VITE_API_URL)
    ? import.meta.env.VITE_API_URL.replace(/\/$/, '')
    : sameOriginApiBase();

/**
 * Returns true when a stored/entered value looks like a browser page URL
 * rather than a bare API base — e.g. it contains `?api=` or `?backend=`.
 * These must never be used as the API base URL.
 */
function looksLikePageUrl(url) {
  if (!url) return false;
  try {
    const u = new URL(url);
    return u.search.includes('api=') || u.search.includes('backend=');
  } catch {
    return false;
  }
}

function resolveInitialApiUrl() {
  if (typeof window !== 'undefined') {
    try {
      const params = new URLSearchParams(window.location.search);
      const urlParam = params.get('api') || params.get('backend');
      if (urlParam) {
        const clean = urlParam.trim().replace(/\/$/, '');
        setStored('float_api_url', clean);
        // Remove the ?api= param from the browser URL so it never leaks into
        // subsequent candidate generation or loopback-fallback logic.
        try { window.history.replaceState({}, '', window.location.pathname); } catch { /* ignore */ }
        return clean;
      }
    } catch {
      /* ignore */
    }
  }
  const stored = getStored('float_api_url');
  // Reject a stored value that looks like a full page URL (e.g. the user
  // accidentally pasted http://localhost:3000/?api=https://... into the
  // Config modal). Fall back to the default so the UI isn't permanently stuck.
  if (stored && looksLikePageUrl(stored)) {
    removeStored('float_api_url');
    return DEFAULT_API_URL;
  }
  // Drop a stale loopback URL when the page is served from a public host —
  // it can only ever point back at the user's own machine.
  if (stored && isLoopbackUrl(stored) && !isPageOnLoopback()) {
    removeStored('float_api_url');
    return DEFAULT_API_URL;
  }
  return stored || DEFAULT_API_URL;
}

let currentApiUrl = resolveInitialApiUrl();

export function getApiUrl() {
  return currentApiUrl;
}

/** Join a backend path onto an API base, preserving any base path prefix. */
function joinApiBase(base, path) {
  try {
    const api = new URL(base, pageOrigin());
    const prefix = api.pathname.replace(/\/$/, '');
    return `${api.origin}${prefix}${path.startsWith('/') ? path : `/${path}`}`;
  } catch {
    return `${base}${path}`;
  }
}

/**
 * Rewrite a backend-returned absolute URL to the currently adopted API base.
 * The backend advertises `http://localhost:8000/...` no matter how the browser
 * reached us; rerouting loopback URLs through the API base keeps them
 * reachable — and same-origin via the proxy — for remote clients.
 */
export function alignUrlToApi(url) {
  if (!url) return url;
  try {
    const target = new URL(url, pageOrigin());
    if (isLoopbackHost(target.hostname)) {
      return joinApiBase(currentApiUrl, `${target.pathname}${target.search}`);
    }
  } catch {
    // Not an absolute URL — leave untouched.
  }
  return url;
}

export function setApiUrl(url) {
  currentApiUrl = url.replace(/\/$/, '');
  setStored('float_api_url', currentApiUrl);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('float:api_url_changed', { detail: currentApiUrl }));
  }
}

/** Forget any stored override and fall back to the default (proxy) base. */
export function resetApiUrl() {
  removeStored('float_api_url');
  currentApiUrl = DEFAULT_API_URL;
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('float:api_url_changed', { detail: currentApiUrl }));
  }
  return currentApiUrl;
}

/** True when the active base is the default (no manual override stored). */
export function isDefaultApiUrl() {
  return currentApiUrl === DEFAULT_API_URL;
}

/**
 * Probe an arbitrary endpoint's /health WITHOUT adopting it, so the Config
 * panel can validate a URL before committing it. Resolves with the health
 * payload, or throws a human-readable Error.
 */
export async function testApiUrl(base) {
  const normalized = String(base || '').trim().replace(/\/$/, '');
  if (!normalized) throw new Error('Enter a URL, or press Reset to use the default.');

  let parsed;
  try {
    parsed = new URL(normalized, pageOrigin());
  } catch {
    throw new Error('That is not a valid URL.');
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('Only http:// or https:// endpoints are supported.');
  }

  let res;
  try {
    res = await fetchWithTimeout(`${normalized}/health`, {}, 8000);
  } catch (err) {
    const timedOut = err.name === 'TimeoutError' || err.name === 'AbortError';
    throw new Error(
      timedOut ? 'Timed out — no response in 8s.' : `Could not connect (${err.message}).`
    );
  }
  if (!res.ok) throw new Error(`Endpoint replied HTTP ${res.status}.`);
  return await res.json().catch(() => ({}));
}

/**
 * Alternate base URLs to try when the configured one is unreachable.
 * Only rewrites loopback-ish hosts (localhost / 127.0.0.1 / 0.0.0.0) —
 * a custom LAN IP or domain is left alone. Port and protocol preserved.
 */
export function getApiCandidates() {
  // Guard: if currentApiUrl somehow contains ?api= (page URL leaked in),
  // reset it so we don't generate broken loopback candidates.
  if (looksLikePageUrl(currentApiUrl)) {
    currentApiUrl = DEFAULT_API_URL;
    removeStored('float_api_url');
  }
  const ordered = [currentApiUrl];
  try {
    const u = new URL(currentApiUrl, pageOrigin());
    if (isLoopbackHost(u.hostname)) {
      for (const host of LOOPBACK_HOSTS) {
        if (host === u.hostname) continue;
        const alt = new URL(u.toString());
        alt.hostname = host;
        ordered.push(alt.toString().replace(/\/$/, ''));
      }
    }
  } catch {
    // Malformed URL — just try it as-is.
  }
  return [...new Set(ordered)];
}

/**
 * Probe GET /health across candidates, adopt the first base that answers,
 * and return it. Throws if none respond.
 */
async function ensureBackend() {
  const errors = [];
  for (const base of getApiCandidates()) {
    try {
      const res = await fetchWithTimeout(`${base}/health`);
      if (res.ok) {
        if (base !== currentApiUrl) {
          console.info(`[api] backend reachable at ${base} (was ${currentApiUrl}) — switching`);
          setApiUrl(base);
        }
        return base;
      }
      errors.push(`${base}: HTTP ${res.status}`);
    } catch (err) {
      errors.push(`${base}: ${err.name === 'TimeoutError' || err.name === 'AbortError' ? 'timeout' : err.message}`);
    }
  }
  console.error(`[api] backend unreachable on all candidates: ${errors.join(' | ')}`);
  throw new Error(
    `Cannot reach backend at ${currentApiUrl} (tried: ${getApiCandidates().join(', ')}) — is it running?`
  );
}

/** Convert a Blob or File to a Base64 string (without the data URL header). */
export function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    if (!blob) {
      reject(new Error('No file selected.'));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === 'string' ? reader.result : '';
      const comma = result.indexOf(',');
      if (comma < 0 || comma === result.length - 1) {
        reject(new Error('Could not read this file.'));
        return;
      }
      resolve(result.slice(comma + 1));
    };
    // `onloadend` also fires after `onerror`; using it to resolve would throw
    // on `null` and leave the caller's promise pending forever.
    reader.onerror = () => reject(reader.error || new Error('Could not read this file.'));
    reader.onabort = () => reject(new Error('Reading the file was cancelled.'));
    reader.readAsDataURL(blob);
  });
}

/** Lower-case, parameter-free MIME type (`audio/webm;codecs=opus` -> `audio/webm`). */
function baseMime(value) {
  return String(value || '').split(';')[0].trim().toLowerCase();
}

function extensionOf(name) {
  const match = /\.([a-z0-9]+)$/i.exec(String(name || ''));
  return match ? match[1].toLowerCase() : '';
}

const IMAGE_MIME_FORMATS = {
  'image/jpeg': 'jpg',
  'image/jpg': 'jpg',
  'image/pjpeg': 'jpg',
  'image/png': 'png',
  'image/webp': 'webp',
};

const IMAGE_EXTENSION_FORMATS = {
  jpg: 'jpg',
  jpeg: 'jpg',
  png: 'png',
  webp: 'webp',
};

// Only the containers the backend can decode. Extensions are mapped to the
// container name pydub/ffmpeg understands (m4a -> mp4, opus -> ogg).
const AUDIO_MIME_FORMATS = {
  'audio/wav': 'wav',
  'audio/wave': 'wav',
  'audio/x-wav': 'wav',
  'audio/vnd.wave': 'wav',
  'audio/mpeg': 'mp3',
  'audio/mp3': 'mp3',
  'audio/mpga': 'mp3',
  'audio/mp4': 'mp4',
  'audio/x-m4a': 'mp4',
  'audio/m4a': 'mp4',
  'audio/ogg': 'ogg',
  'audio/opus': 'ogg',
  'audio/webm': 'webm',
  'audio/flac': 'flac',
  'audio/x-flac': 'flac',
  'audio/aac': 'aac',
};

const AUDIO_EXTENSION_FORMATS = {
  wav: 'wav',
  wave: 'wav',
  mp3: 'mp3',
  mpga: 'mp3',
  m4a: 'mp4',
  mp4: 'mp4',
  ogg: 'ogg',
  oga: 'ogg',
  opus: 'ogg',
  webm: 'webm',
  flac: 'flac',
  aac: 'aac',
};

/**
 * Container hint for `image_format` / `image_base64`.
 *
 * Derive it from the MIME type first, then from the extension. `null` means
 * the file is not an image the backend supports (BMP/GIF/SVG/TIFF are not).
 */
export function imageFormatFromFile(file) {
  const mime = baseMime(file?.type);
  if (IMAGE_MIME_FORMATS[mime]) return IMAGE_MIME_FORMATS[mime];
  return IMAGE_EXTENSION_FORMATS[extensionOf(file?.name)] || null;
}

/** Container hint for `audio_format`; `null` when the file is not decodable. */
export function audioFormatFromFile(file) {
  const mime = baseMime(file?.type);
  if (AUDIO_MIME_FORMATS[mime]) return AUDIO_MIME_FORMATS[mime];
  return AUDIO_EXTENSION_FORMATS[extensionOf(file?.name)] || null;
}

/** Human-readable size for validation messages. */
export function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

/** Log API failures so they are visible in the browser console, not silent. */
function logApiError(method, path, err, status) {
  const url = `${currentApiUrl}${path}`;
  if (err) {
    console.error(`[api] ${method} ${url} — network error: ${err.message}. Is the backend running?`);
  } else {
    console.error(`[api] ${method} ${url} — HTTP ${status}`);
  }
}

/** Check health and model status (tries localhost variants as fallback) */
export async function fetchHealth() {
  const errors = [];
  for (const base of getApiCandidates()) {
    try {
      const res = await fetchWithTimeout(`${base}/health`);
      if (!res.ok) {
        logApiError('GET', '/health', null, res.status);
        errors.push(`${base}: HTTP ${res.status}`);
        continue;
      }
      if (base !== currentApiUrl) {
        console.info(`[api] backend reachable at ${base} (was ${currentApiUrl}) — switching`);
        setApiUrl(base);
      }
      return await res.json();
    } catch (err) {
      errors.push(`${base}: ${err.name === 'TimeoutError' || err.name === 'AbortError' ? 'timeout' : err.message}`);
    }
  }
  console.error(`[api] GET /health failed on all candidates: ${errors.join(' | ')}`);
  throw new Error(`Cannot reach backend (tried: ${getApiCandidates().join(', ')}) — is it running?`);
}

/** Poll generation progress for a request id (chunk-by-chunk). Best-effort. */
export async function fetchProgress(requestId) {
  if (!requestId) return null;
  try {
    const res = await fetchWithTimeout(
      `${currentApiUrl}/progress/${encodeURIComponent(requestId)}`,
      {},
      5000
    );
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** Get live GPU telemetry. A missing endpoint (no CUDA) is not an error. */
export async function fetchGpuMetrics() {
  try {
    return await fetchJson(`${currentApiUrl}/gpu`, {}, { timeoutMs: 8000 });
  } catch (err) {
    // 404/503 simply mean this deployment has no GPU endpoint; the widgets
    // hide themselves and the shared poller must not spam the console.
    const expected = err instanceof HttpError && (err.status === 404 || err.status === 503);
    if (!expected) logApiError('GET', '/gpu', err instanceof HttpError ? null : err, err.status);
    throw err;
  }
}

/** Clear GPU CUDA memory cache */
export async function clearGpuMemory() {
  try {
    return await fetchJson(
      `${currentApiUrl}/gpu/clear`,
      { method: 'POST' },
      { timeoutMs: 10000 }
    );
  } catch (err) {
    logApiError('POST', '/gpu/clear', err instanceof HttpError ? null : err, err.status);
    throw err;
  }
}

/**
 * Fetch the perf matrix: the speed/throughput + latency benchmark result.
 * Returns { status, matrix, error, started_at, finished_at } or null when the
 * backend does not answer (the panel then shows "no measurement yet").
 */
export async function fetchPerfMatrix() {
  try {
    return await fetchJson(`${currentApiUrl}/perf/matrix`, {}, { timeoutMs: 8000 });
  } catch (err) {
    logApiError('GET', '/perf/matrix', err instanceof HttpError ? null : err, err.status);
    return null;
  }
}

/**
 * Fetch the rolling per-request view: distributions (p50/p95/p99) across the
 * recent requests, plus the last few samples. Returns null when the backend
 * does not answer.
 */
export async function fetchPerfRequests() {
  try {
    return await fetchJson(`${currentApiUrl}/perf/requests`, {}, { timeoutMs: 8000 });
  } catch (err) {
    logApiError('GET', '/perf/requests', err instanceof HttpError ? null : err, err.status);
    return null;
  }
}

/**
 * Start a benchmark run on the backend. It holds the GPU (live requests queue
 * behind it), so the panel warns before the call. Returns the initial status.
 */
export async function startPerfBenchmark(options = {}) {
  try {
    return await fetchJson(
      `${currentApiUrl}/perf/benchmark`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(options),
      },
      { timeoutMs: 15000 }
    );
  } catch (err) {
    logApiError('POST', '/perf/benchmark', err instanceof HttpError ? null : err, err.status);
    if (err instanceof HttpError) throw err;
    throw new Error(`Cannot reach backend at ${currentApiUrl} — is it running?`);
  }
}

/** Generate talking portrait video */
export async function generateVideo(payload) {
  // Probe first and send the POST exactly once to the reachable base —
  // blindly retrying a POST across hosts could submit duplicate GPU jobs.
  await ensureBackend();
  let res;
  try {
    res = await fetch(`${currentApiUrl}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    logApiError('POST', '/generate', err);
    throw new Error(`Cannot reach backend at ${currentApiUrl} — is it running?`);
  }

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = errorData.detail || 'Video generation failed';
    console.error(`[api] POST ${currentApiUrl}/generate — HTTP ${res.status}: ${detail}`);
    const prefix = res.status === 429 ? 'Server busy — ' : res.status === 400 ? 'Bad request — ' : '';
    throw new Error(`${prefix}${detail}`);
  }

  return await res.json();
}

/**
 * Server-provided STUN servers (fed from /livekit/status -> stun_urls, which
 * reads fast_float/config/settings.toml [webrtc]). When the deployment ships
 * no STUN list, the built-in public Google servers keep ICE working.
 */
let serverIceServers = null;

/**
 * Rewrite an ICE URL whose host is loopback to the host the browser actually
 * uses for the API. The backend advertises itself (e.g. `stun:localhost:3478`)
 * but a remote browser cannot gather candidates from its own machine — and
 * when the API sits behind the Vite/prod proxy the request Host header is
 * often loopback. Explicit operator hosts (coturn, a public DNS name) are left
 * untouched. Both STUN and TURN schemes are supported.
 */
function alignIceUrl(url) {
  const value = String(url || '').trim();
  const match = /^(stuns?|turns?):(\[[^\]]+\]|[^:?]+)(?::(\d+))?([^ ]*)$/i.exec(value);
  if (!match) return value;
  const scheme = match[1].toLowerCase();
  let host = match[2].replace(/^\[|\]$/g, '');
  if (isLoopbackHost(host)) {
    try {
      const apiHost = new URL(currentApiUrl, pageOrigin()).hostname;
      if (apiHost && !isLoopbackHost(apiHost)) host = apiHost;
    } catch {
      /* keep the loopback host */
    }
  }
  const port = match[3] ? `:${match[3]}` : '';
  return `${scheme}:${host.includes(':') ? `[${host}]` : host}${port}${match[4] || ''}`;
}

function normalizeIceServer(entry) {
  if (typeof entry === 'string') return { urls: alignIceUrl(entry) };
  if (!entry || typeof entry !== 'object') return null;
  const rawUrls = Array.isArray(entry.urls) ? entry.urls : [entry.urls];
  const urls = rawUrls.filter(usableIceUrl).map(alignIceUrl);
  if (!urls.length) return null;
  const server = { urls: urls.length === 1 ? urls[0] : urls };
  if (entry.username) server.username = entry.username;
  if (entry.credential) server.credential = entry.credential;
  return server;
}

/**
 * Drop a STUN target the browser provably cannot reach. The page's own HTTPS
 * origin is served by an HTTP(S)-only tunnel (cloudflared/ngrok): it never
 * carries UDP, so a server that advertises its loopback STUN responder gets
 * rewritten by alignIceUrl() to the tunnel host — worthless for gathering, but
 * worse, it would REPLACE the public Google fallback when it is the sole
 * entry, leaving the browser without any working STUN (blank video, no audio
 * over LiveKit and the raw WebRTC fallback alike). TURN is kept: a TURN relay
 * on the page's own host can legitimately listen on TCP/TLS 443.
 */
function usableIceUrl(url) {
  const value = String(url || '').trim();
  if (!value) return false;
  const match = /^(stuns?):(\[[^\]]+\]|[^:?]+)/i.exec(value);
  if (!match) return true; // TURN or an unparsable URL: leave the judgement up
  if (typeof window === 'undefined' || window.location.protocol !== 'https:') return true;
  return match[2].replace(/^\[|\]$/g, '').toLowerCase() !== window.location.hostname.toLowerCase();
}

/**
 * Feed the raw WebRTC transport an ICE list from `/livekit/status`.
 * Accepts either `ice_servers` objects (`{urls, username, credential}`) or a
 * plain STUN URL list for compatibility. An explicit empty list is kept as
 * empty so a deployment that disabled ICE can opt out of the Google default.
 */
/**
 * Feed the raw WebRTC transport an ICE list from `/livekit/status`.
 * Accepts either `ice_servers` objects (`{urls, username, credential}`) or a
 * plain STUN URL list for compatibility. An explicit empty list is kept as
 * empty so a deployment that disabled ICE can opt out of the Google default.
 * Entries that are unusable from the page (loopback STUN behind an HTTPS
 * tunnel — see usableIceUrl) are dropped instead; when the server *provided*
 * entries but none of them is usable, the list falls back to null so the
 * public Google default survives.
 */
export function setIceServers(servers) {
  if (!Array.isArray(servers)) {
    serverIceServers = null;
    return;
  }
  const mapped = servers.map(normalizeIceServer).filter(Boolean);
  serverIceServers = servers.length && !mapped.length ? null : mapped;
}

export function getIceServers() {
  return serverIceServers ?? [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ];
}

/**
 * Attach a remote track to a <video>, merging tracks into one MediaStream.
 *
 * aiortc gives every sender its own stream id, so audio and video arrive in
 * separate MediaStream objects. Assigning `event.streams[0]` on each ontrack
 * would replace the previous one and drop whichever track came first.
 *
 * There is at most one track per kind: when a new request reuses the element,
 * the previous session's track must be replaced, not left alongside the new
 * one. A MediaStream with two video tracks renders whichever the browser
 * picks first, which shows up as a frozen preview on the old track.
 */
export function attachRemoteTrack(videoEl, event) {
  if (!videoEl) return null;
  let media = videoEl.srcObject;
  if (!(media instanceof MediaStream)) {
    media = new MediaStream();
    videoEl.srcObject = media;
  }
  const streams = event.streams ?? [];
  const incoming = streams.length ? streams.flatMap((s) => s.getTracks()) : [event.track];
  for (const track of incoming) {
    if (!track) continue;
    for (const existing of media.getTracks()) {
      if (existing.kind === track.kind && existing.id !== track.id) {
        media.removeTrack(existing);
      }
    }
    if (!media.getTracks().some((t) => t.id === track.id)) media.addTrack(track);
  }
  return media;
}

/** Establish WebRTC peer connection using /offer */
export async function createWebRtcSession(streamId, onTrackCallback, onStateCallback, options = {}) {
  const { live = false } = options;
  const pc = new RTCPeerConnection({
    // Server-provided STUN/TURN list (see /livekit/status -> ice_servers)
    // replaces the public Google default when present.
    iceServers: getIceServers(),
  });

  // Recv-only viewer: declare the m-lines BEFORE createOffer. Without these
  // the offer SDP has zero media sections and the server answers with
  // nothing to play (black video, ontrack never fires).
  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.addTransceiver('audio', { direction: 'recvonly' });

  if (onTrackCallback) {
    pc.ontrack = onTrackCallback;
  }
  if (onStateCallback) {
    pc.onconnectionstatechange = () => onStateCallback(pc.connectionState);
  }
  // Surface STUN/TURN failures in the browser console; a TCP-only proxy
  // produces candidate errors here even though SDP negotiation succeeds.
  pc.onicecandidateerror = (event) => {
    console.warn(
      `[webrtc] ICE candidate error ${event.errorCode ?? '?'}: ${event.errorText || 'unknown'} (${event.url || 'no url'})`
    );
  };

  // Trickle ICE. `/offer` only carries the initial SDP; candidates discovered
  // afterwards (especially TURN/TCP allocations) are POSTed to
  // /offer/candidate using the peer id returned with the answer.
  let peerId = null;
  let closed = false;
  let endOfCandidatesPending = false;
  const pendingCandidates = [];
  let candidateChain = Promise.resolve();
  const postCandidate = (candidate) => {
    if (closed) return;
    const payload = {
      peer_id: peerId,
      candidate: candidate ? candidate.candidate : null,
      sdp_mid: candidate ? candidate.sdpMid : null,
      sdp_mline_index: candidate ? candidate.sdpMLineIndex : null,
    };
    candidateChain = candidateChain
      .then(() =>
        fetch(`${currentApiUrl}/offer/candidate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })
      )
      .then((candidateRes) => {
        if (!candidateRes.ok) {
          console.warn(`[webrtc] candidate rejected (HTTP ${candidateRes.status})`);
        }
      })
      .catch((err) => {
        if (!closed) console.warn('[webrtc] candidate post failed:', err);
      });
  };
  pc.onicecandidate = (event) => {
    if (closed) return;
    if (event.candidate) {
      if (peerId) postCandidate(event.candidate);
      else pendingCandidates.push(event.candidate);
    } else if (peerId) {
      postCandidate(null);
    } else {
      // Gathering finished before /offer returned; flush after peer_id.
      endOfCandidatesPending = true;
    }
  };

  // Create local offer
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // The browser appends `a=end-of-candidates` once gathering completes. If
  // that line is in the offer, aioice closes its remote-candidate list before
  // later TURN candidates arrive. Omit it and send an explicit null candidate
  // via /offer/candidate instead (after peer_id is known).
  const offerBody = () =>
    JSON.stringify({
      sdp: pc.localDescription.sdp.replace(/^a=end-of-candidates\r?\n/gm, ''),
      type: pc.localDescription.type,
      stream_id: streamId,
      live,
    });

  const res = await fetch(`${currentApiUrl}/offer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: offerBody(),
  }).catch(async (err) => {
    // Single retry path: find a reachable base first, then send once.
    logApiError('POST', '/offer', err);
    await ensureBackend();
    return fetch(`${currentApiUrl}/offer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: offerBody(),
    });
  }).catch((err2) => {
    closed = true;
    pc.close();
    throw new Error(`Cannot reach backend at ${currentApiUrl} — is it running? (${err2.message})`);
  });

  if (!res.ok) {
    closed = true;
    pc.close();
    const errorData = await res.json().catch(() => ({}));
    const detail = errorData.detail || res.statusText;
    console.error(`[api] POST ${currentApiUrl}/offer — HTTP ${res.status}: ${detail}`);
    throw new Error(`WebRTC negotiation failed (${res.status}): ${detail}`);
  }

  const answer = await res.json();
  peerId = answer.peer_id || null;
  if (peerId) {
    // Flush candidates that were gathered while /offer was in flight.
    for (const candidate of pendingCandidates.splice(0)) postCandidate(candidate);
    if (endOfCandidatesPending) postCandidate(null);
  } else {
    console.warn('[webrtc] backend did not return a peer_id; late ICE candidates are dropped');
  }

  try {
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
  } catch (err) {
    closed = true;
    pc.close();
    throw err;
  }

  return pc;
}
