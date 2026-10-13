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

function isLoopbackUrl(value) {
  try {
    return isLoopbackHost(new URL(value, window.location.origin).hostname);
  } catch {
    return false;
  }
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

function resolveInitialApiUrl() {
  const stored = localStorage.getItem('float_api_url');
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
}

/**
 * Alternate base URLs to try when the configured one is unreachable.
 * Only rewrites loopback-ish hosts (localhost / 127.0.0.1 / 0.0.0.0) —
 * a custom LAN IP or domain is left alone. Port and protocol preserved.
 */
export function getApiCandidates() {
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

/** Establish WebRTC peer connection using /offer */
export async function createWebRtcSession(streamId, onTrackCallback) {
  const pc = new RTCPeerConnection({
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun1.l.google.com:19302' },
    ],
  });

  if (onTrackCallback) {
    pc.ontrack = onTrackCallback;
  }

  // Create local offer
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const res = await fetch(`${currentApiUrl}/offer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      stream_id: streamId,
    }),
  }).catch(async (err) => {
    // Single retry path: find a reachable base first, then send once.
    logApiError('POST', '/offer', err);
    await ensureBackend();
    return fetch(`${currentApiUrl}/offer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
        stream_id: streamId,
      }),
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
