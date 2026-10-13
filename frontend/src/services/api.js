/**
 * FLOAT Studio API Client
 */

export const DEFAULT_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

let currentApiUrl = localStorage.getItem('float_api_url') || DEFAULT_API_URL;

export function getApiUrl() {
  return currentApiUrl;
}

export function setApiUrl(url) {
  currentApiUrl = url.replace(/\/$/, '');
  localStorage.setItem('float_api_url', currentApiUrl);
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

/** Check health and model status */
export async function fetchHealth() {
  let res;
  try {
    res = await fetch(`${currentApiUrl}/health`);
  } catch (err) {
    logApiError('GET', '/health', err);
    throw new Error(`Cannot reach backend at ${currentApiUrl} — is it running?`);
  }
  if (!res.ok) {
    logApiError('GET', '/health', null, res.status);
    throw new Error(`Health check failed: ${res.status} ${res.statusText}`);
  }
  return await res.json();
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
  }).catch((err) => {
    pc.close();
    logApiError('POST', '/offer', err);
    throw new Error(`Cannot reach backend at ${currentApiUrl} — is it running?`);
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
