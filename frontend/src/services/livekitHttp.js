/**
 * HTTP-only helpers for the LiveKit integration.
 *
 * These talk to the FLOAT backend (`/livekit/*`). They are kept separate from
 * `livekit.js` on purpose: importing this module must never pull the
 * `livekit-client` SDK (≈1.4 MB source) into the initial Studio bundle.
 * Components that only need status/token/conversation calls import from here.
 */

import { getApiUrl } from './api';
import { fetchJson, HttpError } from './http';
import { getStored, setStored } from './storage';

/** LocalStorage key for a user-configured LiveKit SFU URL override. */
const LIVEKIT_STORAGE_KEY = 'float_livekit_url';

/**
 * LocalStorage key for the optional Gemini API key the Studio sends per
 * session. Stored in the browser so the operator does not have to edit
 * server config to try the Gemini Live engine; the server-side
 * secerts.toml ``[gemini].api_key`` is still the recommended persistent
 * location.
 */
const GEMINI_API_KEY_STORAGE_KEY = 'float_gemini_api_key';

/**
 * LocalStorage key for the optional Gemini Live model name the Studio sends
 * per session; empty means "use the server's default (plugin) model".
 */
const GEMINI_MODEL_STORAGE_KEY = 'float_gemini_model';

export function getGeminiApiKey() {
  return getStored(GEMINI_API_KEY_STORAGE_KEY, '') || '';
}

export function setGeminiApiKey(key) {
  const clean = (key || '').trim();
  if (clean) setStored(GEMINI_API_KEY_STORAGE_KEY, clean);
  else setStored(GEMINI_API_KEY_STORAGE_KEY, '');
}

export function getGeminiModel() {
  return getStored(GEMINI_MODEL_STORAGE_KEY, '') || '';
}

export function setGeminiModel(model) {
  setStored(GEMINI_MODEL_STORAGE_KEY, (model || '').trim());
}

export function getLiveKitUrlOverride() {
  return getStored(LIVEKIT_STORAGE_KEY, '') || '';
}

export function setLiveKitUrlOverride(url) {
  setStored(LIVEKIT_STORAGE_KEY, (url || '').trim());
}

/**
 * Align the LiveKit server URL to the browser's context.
 *
 * If the server reports a loopback address (127.0.0.1 or localhost) but the
 * client is visiting from a remote LAN IP or domain, rewrite the host to
 * window.location.hostname. Also upgrades ws: to wss: on HTTPS pages to avoid
 * mixed-content blocks.
 */
export function alignLiveKitUrl(url) {
  if (!url) return url;
  try {
    const target = new URL(url);
    const loopbacks = ['127.0.0.1', 'localhost', '0.0.0.0'];
    if (loopbacks.includes(target.hostname)) {
      if (
        typeof window !== 'undefined'
        && window.location.hostname
        && !loopbacks.includes(window.location.hostname)
      ) {
        target.hostname = window.location.hostname;
      }
    }
    if (
      typeof window !== 'undefined'
      && window.location.protocol === 'https:'
      && target.protocol === 'ws:'
    ) {
      target.protocol = 'wss:';
    }
    return target.toString();
  } catch {
    return url;
  }
}

/**
 * Fetch the current LiveKit server configuration status.
 *
 * `verify: true` dials the SFU once (mints a token + connects) and refreshes
 * the backend's connectivity check, so the Config panel can prove credentials
 * work before a session is started. Verification gets a longer deadline.
 *
 * Never throws: a dead backend is reported as `{ reachable: false }` so the
 * status pills and the readiness UI have something to render.
 */
export async function fetchLiveKitStatus({ verify = false, timeoutMs } = {}) {
  const deadline = timeoutMs ?? (verify ? 25000 : 8000);
  try {
    const suffix = verify ? '?verify=1' : '';
    const data = await fetchJson(
      `${getApiUrl()}/livekit/status${suffix}`,
      {},
      { timeoutMs: deadline }
    );
    return { ...(data || {}), reachable: true };
  } catch (err) {
    if (err instanceof HttpError) {
      return {
        configured: false,
        reachable: true,
        warnings: [`LiveKit status request failed (HTTP ${err.status}).`],
      };
    }
    return {
      configured: false,
      reachable: false,
      warnings: [err?.message || 'Cannot reach the backend'],
    };
  }
}

/**
 * Mint a viewer token for a room.
 *
 * `publishMicrophone` opts into a publish grant so the browser can send its
 * microphone into the room. The Gemini Live backend listens to that track
 * instead of receiving PCM over HTTP.
 */
export async function fetchLiveKitToken(room, name = 'FLOAT Viewer', { publishMicrophone = false } = {}) {
  return fetchJson(
    `${getApiUrl()}/livekit/token`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ room, name, publish_mic: publishMicrophone }),
    },
    { timeoutMs: 12000 }
  );
}

/** Trigger backend publishing of an active or completed stream to a room. */
export async function triggerLiveKitPublish(requestId, room = null) {
  return fetchJson(
    `${getApiUrl()}/livekit/publish/${encodeURIComponent(requestId)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ room }),
    },
    { timeoutMs: 12000 }
  );
}

/**
 * Whether the backend publisher for a request is running, and its last error.
 * Publishing is async, so this explains why a room never received tracks.
 * Returns null when unavailable.
 */
export async function fetchLiveKitPublishStatus(requestId) {
  try {
    return await fetchJson(
      `${getApiUrl()}/livekit/publish/${encodeURIComponent(requestId)}`,
      {},
      { timeoutMs: 5000 }
    );
  } catch {
    return null;
  }
}

/**
 * Start a live-audio session (agent + renderer + avatar worker).
 *
 * Warming up the renderer can take a while, hence the generous deadline; the
 * caller can still pass its own AbortSignal.
 */
export async function startConversation({
  room = null,
  imageBase64 = null,
  imageFormat = 'jpg',
  agent = 'builtin',
  agentName = null,
  geminiApiKey = null,
  geminiModel = null,
  signal,
} = {}) {
  return fetchJson(
    `${getApiUrl()}/livekit/conversation/start`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        room,
        image_base64: imageBase64,
        image_format: imageFormat,
        agent,
        agent_name: agentName,
        gemini_api_key: geminiApiKey || null,
        gemini_model: geminiModel || null,
      }),
      ...(signal ? { signal } : {}),
    },
    { timeoutMs: 30000 }
  );
}

/** Push one ~200 ms chunk of mono int16 PCM (16 kHz) to the avatar renderer. */
export async function pushConversationAudio(room, audioBase64, { flush = false } = {}) {
  return fetchJson(
    `${getApiUrl()}/livekit/conversation/audio`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ room, audio_base64: audioBase64, sample_rate: 16000, flush }),
    },
    { timeoutMs: 8000 }
  );
}

/** Barge-in: stop the avatar's current utterance immediately. */
export async function interruptConversation(room) {
  try {
    return await fetchJson(
      `${getApiUrl()}/livekit/conversation/interrupt`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ room }),
      },
      { timeoutMs: 5000 }
    );
  } catch {
    return null;
  }
}

/** End the live session and stop its renderer/worker. */
export async function stopConversation(room) {
  try {
    return await fetchJson(
      `${getApiUrl()}/livekit/conversation/stop`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ room }),
      },
      { timeoutMs: 10000 }
    );
  } catch {
    return null;
  }
}

/**
 * Whether the conversation's agent participant has actually joined the room.
 *
 * Reaching 'Live' only proves the avatar's tracks exist; in Gemini Live mode
 * the stage stays silent until the worker joins. Returns null when the room is
 * gone or the backend is unreachable.
 */
export async function fetchConversationStatus(room) {
  try {
    return await fetchJson(
      `${getApiUrl()}/livekit/conversation/status?room=${encodeURIComponent(room)}`,
      {},
      { timeoutMs: 5000 }
    );
  } catch {
    return null;
  }
}
