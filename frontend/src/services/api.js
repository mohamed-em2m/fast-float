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

/** Check health and model status */
export async function fetchHealth() {
  const res = await fetch(`${currentApiUrl}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return await res.json();
}

/** Get live GPU telemetry */
export async function fetchGpuMetrics() {
  const res = await fetch(`${currentApiUrl}/gpu`);
  if (!res.ok) throw new Error(`GPU metrics failed: ${res.statusText}`);
  return await res.json();
}

/** Clear GPU CUDA memory cache */
export async function clearGpuMemory() {
  const res = await fetch(`${currentApiUrl}/gpu/clear`, { method: 'POST' });
  if (!res.ok) throw new Error(`Clear GPU cache failed: ${res.statusText}`);
  return await res.json();
}

/** Generate talking portrait video */
export async function generateVideo(payload) {
  const res = await fetch(`${currentApiUrl}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || 'Video generation failed');
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
  });

  if (!res.ok) {
    pc.close();
    throw new Error(`WebRTC negotiation failed: ${res.statusText}`);
  }

  const answer = await res.json();
  await pc.setRemoteDescription(new RTCSessionDescription(answer));

  return pc;
}
