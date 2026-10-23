/**
 * FLOAT Studio API Client
 *
 * The backend is reached through the same-origin `/api` reverse proxy (see
 * vite.config.js), so the browser talks to whatever host served the page —
 * localhost, a LAN IP, or an ngrok tunnel. Hard-coding `http://localhost:8000`
 * breaks every remote browser: there, `localhost` is the *user's* machine.
 */

/** Path prefix proxied to the FastAPI backend by Vite (and any prod proxy). */
const API_PROXY_PATH = '/api';

/** Hosts that mean "this very machine" and are useless to a remote browser. */
const LOOPBACK_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0'];

function isLoopbackHost(hostname) {
  return LOOPBACK_HOSTS.includes(hostname);
}

export function isLoopbackUrl(value) {
  try {
    return isLoopbackHost(new URL(value, window.location.origin).hostname);
  } catch {
    return false;
  }
}

/** True when the browser tab itself is served from a loopback host. */
export function isPageOnLoopback() {
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
        localStorage.setItem('float_api_url', clean);
        // Remove the ?api= param from the browser URL so it never leaks into
        // subsequent candidate generation or loopback-fallback logic.
        try { window.history.replaceState({}, '', window.location.pathname); } catch { /* ignore */ }
        return clean;
      }
    } catch {
      /* ignore */
    }
  }
  const stored = localStorage.getItem('float_api_url');
  // Reject a stored value that looks like a full page URL (e.g. the user
  // accidentally pasted http://localhost:3000/?api=https://... into the
  // Config modal). Fall back to the default so the UI isn't permanently stuck.
  if (stored && looksLikePageUrl(stored)) {
    localStorage.removeItem('float_api_url');
    return DEFAULT_API_URL;
  }
  // Drop a stale loopback URL when the page is served from a public host —
  // it can only ever point back at the user's own machine.
  if (stored && isLoopbackUrl(stored) && !isLoopbackHost(window.location.hostname)) {
    localStorage.removeItem('float_api_url');
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
    const api = new URL(base, window.location.origin);
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
    const target = new URL(url, window.location.origin);
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
  localStorage.setItem('float_api_url', currentApiUrl);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('float:api_url_changed', { detail: currentApiUrl }));
  }
}

/** Forget any stored override and fall back to the default (proxy) base. */
export function resetApiUrl() {
  localStorage.removeItem('float_api_url');
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
    parsed = new URL(normalized, window.location.origin);
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
    throw new Error(
      err.name === 'AbortError' ? 'Timed out — no response in 8s.' : `Could not connect (${err.message}).`
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
    localStorage.removeItem('float_api_url');
  }
  const ordered = [currentApiUrl];
  try {
    const u = new URL(currentApiUrl, window.location.origin);
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

/** fetch with a timeout so a dead host fails fast instead of hanging. */
async function fetchWithTimeout(url, init = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
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
      errors.push(`${base}: ${err.name === 'AbortError' ? 'timeout' : err.message}`);
    }
  }
  console.error(`[api] backend unreachable on all candidates: ${errors.join(' | ')}`);
  throw new Error(
    `Cannot reach backend at ${currentApiUrl} (tried: ${getApiCandidates().join(', ')}) — is it running?`
  );
}

/** Convert a Blob or File to a Base64 string (without data URL header) */
export function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result.split(',')[1];
      resolve(base64String);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
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
      errors.push(`${base}: ${err.name === 'AbortError' ? 'timeout' : err.message}`);
    }
  }
  console.error(`[api] GET /health failed on all candidates: ${errors.join(' | ')}`);
  throw new Error(`Cannot reach backend (tried: ${getApiCandidates().join(', ')}) — is it running?`);
}

/** Poll generation progress for a request id (chunk-by-chunk). Best-effort. */
export async function fetchProgress(requestId) {
  if (!requestId) return null;
  try {
    const res = await fetch(`${currentApiUrl}/progress/${encodeURIComponent(requestId)}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** Get live GPU telemetry */
export async function fetchGpuMetrics() {
  let res;
  try {
    res = await fetch(`${currentApiUrl}/gpu`);
  } catch (err) {
    logApiError('GET', '/gpu', err);
    throw new Error(`Cannot reach backend at ${currentApiUrl} — is it running?`);
  }
  if (!res.ok) {
    logApiError('GET', '/gpu', null, res.status);
    throw new Error(`GPU metrics failed: ${res.status} ${res.statusText}`);
  }
  return await res.json();
}

/** Clear GPU CUDA memory cache */
export async function clearGpuMemory() {
  let res;
  try {
    res = await fetch(`${currentApiUrl}/gpu/clear`, { method: 'POST' });
  } catch (err) {
    logApiError('POST', '/gpu/clear', err);
    throw new Error(`Cannot reach backend at ${currentApiUrl} — is it running?`);
  }
  if (!res.ok) {
    logApiError('POST', '/gpu/clear', null, res.status);
    throw new Error(`Clear GPU cache failed: ${res.status} ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Fetch the perf matrix: the speed/throughput + latency benchmark result.
 * Returns { status, matrix, error, started_at, finished_at } or null when the
 * backend does not answer (the panel then shows "no measurement yet").
 */
export async function fetchPerfMatrix() {
  try {
    const res = await fetch(`${currentApiUrl}/perf/matrix`);
    if (!res.ok) {
      logApiError('GET', '/perf/matrix', null, res.status);
      return null;
    }
    return await res.json();
  } catch (err) {
    logApiError('GET', '/perf/matrix', err);
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
    const res = await fetch(`${currentApiUrl}/perf/requests`);
    if (!res.ok) {
      logApiError('GET', '/perf/requests', null, res.status);
      return null;
    }
    return await res.json();
  } catch (err) {
    logApiError('GET', '/perf/requests', err);
    return null;
  }
}

/**
 * Start a benchmark run on the backend. It holds the GPU (live requests queue
 * behind it), so the panel warns before the call. Returns the initial status.
 */
export async function startPerfBenchmark(options = {}) {
  let res;
  try {
    res = await fetch(`${currentApiUrl}/perf/benchmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options),
    });
  } catch (err) {
    logApiError('POST', '/perf/benchmark', err);
    throw new Error(`Cannot reach backend at ${currentApiUrl} — is it running?`);
  }
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    const detail = errorData.detail || `Benchmark failed: ${res.status}`;
    logApiError('POST', '/perf/benchmark', null, res.status);
    throw new Error(detail);
  }
  return await res.json();
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

export function setIceServers(stunUrls) {
  if (Array.isArray(stunUrls) && stunUrls.length > 0) {
    serverIceServers = stunUrls.map((urls) => ({ urls }));
  } else {
    serverIceServers = null;
  }
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
 */
export function attachRemoteTrack(videoEl, event) {
  if (!videoEl) return null;
  let media = videoEl.srcObject;
  // Reuse the element's stream only while it is still live: a stopped stream
  // keeps its ended tracks and can make the element ignore the new ones.
  if (!(media instanceof MediaStream) || !media.getTracks().some((t) => t.readyState === 'live')) {
    media = new MediaStream();
    videoEl.srcObject = media;
  }
  const streams = event.streams ?? [];
  const incoming = streams.length ? streams.flatMap((s) => s.getTracks()) : [event.track];
  for (const track of incoming) {
    if (track && !media.getTracks().some((t) => t.id === track.id)) media.addTrack(track);
  }
  return media;
}

/** Establish WebRTC peer connection using /offer */
export async function createWebRtcSession(streamId, onTrackCallback, onStateCallback, options = {}) {
  const { live = false } = options;
  const pc = new RTCPeerConnection({
    // Server-provided STUN list (see /livekit/status -> stun_urls, ideally the
    // self-hosted coturn) replaces the public Google default when present.
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

  // Create local offer
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const offerBody = () =>
    JSON.stringify({
      sdp: pc.localDescription.sdp,
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
    pc.close();
    throw new Error(`Cannot reach backend at ${currentApiUrl} — is it running? (${err2.message})`);
  });

  if (!res.ok) {
    pc.close();
    const errorData = await res.json().catch(() => ({}));
    const detail = errorData.detail || res.statusText;
    console.error(`[api] POST ${currentApiUrl}/offer — HTTP ${res.status}: ${detail}`);
    throw new Error(`WebRTC negotiation failed (${res.status}): ${detail}`);
  }

  const answer = await res.json();
  await pc.setRemoteDescription(new RTCSessionDescription(answer));

  return pc;
}
