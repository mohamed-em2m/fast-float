/**
 * LiveKit client transport for the live preview and the conversational avatar.
 *
 * Grounded in docs/LIVEKIT_INTEGRATION.md:
 * - joins a room using a short-lived token minted by POST /livekit/token
 * - subscribes to tracks published by the avatar worker (kind agent with lk.publish_on_behalf)
 * - attaches video + audio to the preview <video> element
 */

import { Room, RoomEvent, Track } from 'livekit-client';
import { getApiUrl } from './api';

/**
 * Align LiveKit server URL to the browser's context.
 *
 * If the server reports a loopback address (127.0.0.1 or localhost) but the client
 * is visiting from a remote LAN IP or domain, rewrite the host to window.location.hostname.
 * Also upgrades ws: to wss: when accessed over HTTPS to prevent mixed content blocks.
 */

/**
 * LocalStorage key for user-configured LiveKit SFU URL override.
 */
const LIVEKIT_STORAGE_KEY = 'float_livekit_url';

export function getLiveKitUrlOverride() {
  if (typeof window === 'undefined') return '';
  return localStorage.getItem(LIVEKIT_STORAGE_KEY) || '';
}

export function setLiveKitUrlOverride(url) {
  if (typeof window === 'undefined') return;
  const clean = (url || '').trim();
  if (clean) {
    localStorage.setItem(LIVEKIT_STORAGE_KEY, clean);
  } else {
    localStorage.removeItem(LIVEKIT_STORAGE_KEY);
  }
}

export function alignLiveKitUrl(url) {
  if (!url) return url;
  try {
    const target = new URL(url);
    const loopbacks = ['127.0.0.1', 'localhost', '0.0.0.0'];
    if (loopbacks.includes(target.hostname)) {
      if (typeof window !== 'undefined' && window.location.hostname && !loopbacks.includes(window.location.hostname)) {
        target.hostname = window.location.hostname;
      }
    }
    if (typeof window !== 'undefined' && window.location.protocol === 'https:' && target.protocol === 'ws:') {
      target.protocol = 'wss:';
    }
    return target.toString();
  } catch {
    return url;
  }
}

const CONNECT_TIMEOUT_MS = 15000;
const FIRST_TRACK_TIMEOUT_MS = 25000;

/**
 * Resolve/reject `promise` within `ms`, running `onTimeout` on expiry.
 *
 * livekit-client's connect can wait forever when signalling succeeds but the
 * media path never comes up (blocked UDP/TCP, dead SFU). Without a bound the
 * studio sits on "Connecting…" and never falls back.
 */
function withTimeout(promise, ms, message, onTimeout) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      try {
        onTimeout?.();
      } catch {
        /* ignore cleanup errors */
      }
      reject(new Error(message));
    }, ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/**
 * Fetch current LiveKit server configuration status.
 */
export async function fetchLiveKitStatus() {
  try {
    const res = await fetch(`${getApiUrl()}/livekit/status`);
    if (!res.ok) return { configured: false };
    return await res.json();
  } catch {
    return { configured: false };
  }
}

/**
 * Mint a viewer token for a given room.
 */
export async function fetchLiveKitToken(room, name = 'FLOAT Viewer') {
  const res = await fetch(`${getApiUrl()}/livekit/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ room, name, can_publish: false }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to mint LiveKit token (${res.status})`);
  }

  return await res.json();
}

/**
 * Trigger backend publishing of an active or completed stream to a LiveKit room.
 */
export async function triggerLiveKitPublish(requestId, room = null) {
  const res = await fetch(`${getApiUrl()}/livekit/publish/${requestId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ room }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to trigger LiveKit publishing (${res.status})`);
  }

  return await res.json();
}

/**
 * Whether the backend publisher for a request is running, and its last error.
 * Publishing is async, so this explains why a room never received tracks.
 */
export async function fetchLiveKitPublishStatus(requestId) {
  try {
    const res = await fetch(`${getApiUrl()}/livekit/publish/${encodeURIComponent(requestId)}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/**
 * Connect to a LiveKit room and stream avatar tracks to the video element.
 *
 * Video track is attached directly to `videoElement`.
 * Audio track is attached to a separate hidden <audio> element that is
 * appended to the document body so playback is never blocked by autoplay
 * policies that apply to <video>.
 *
 * @param {Object} options
 * @param {string} options.room - Room name to join
 * @param {HTMLMediaElement} options.videoElement - <video> element to attach the video track to
 * @param {function} [options.onStatus] - Callback for status updates ('connecting', 'reconnecting', 'live', 'done', 'error')
 * @param {number} [options.connectTimeoutMs] - Bound on the SFU handshake
 * @param {number} [options.trackTimeoutMs] - Bound on waiting for the first track
 * @returns {Promise<{ room: Room, disconnect: function }>}
 */
export async function connectLiveKitSession({
  room: roomName,
  videoElement,
  onStatus,
  connectTimeoutMs = CONNECT_TIMEOUT_MS,
  trackTimeoutMs = FIRST_TRACK_TIMEOUT_MS,
}) {
  onStatus?.('connecting');

  // 1. Get token
  const tokenData = await fetchLiveKitToken(roomName);
  const overrideUrl = getLiveKitUrlOverride();
  const livekitUrl = alignLiveKitUrl(overrideUrl || tokenData.url);
  const token = tokenData.token;

  // 2. Create LiveKit Room. LiveKit keeps the ICE/TURN list its own server
  // hands back; the main-server STUN responder is used by the raw WebRTC
  // fallback transport instead.
  const room = new Room({
    adaptiveStream: true,
    dynacast: true,
  });

  const attachedTracks = new Set();
  // One live track per kind: a publisher reconnect republishes the track, and
  // attaching both would leave the old (frozen) video behind the new one.
  const tracksByKind = new Map();
  let failed = false;
  let resolveFirstTrack;
  const firstTrack = new Promise((resolve) => {
    resolveFirstTrack = resolve;
  });

  // Hidden <audio> element for the avatar audio track.
  // Kept separate from <video> to avoid double-playback (video element
  // already decodes audio from the muxed track in some browsers).
  let audioEl = null;
  const getAudioEl = () => {
    if (!audioEl) {
      audioEl = document.createElement('audio');
      audioEl.autoplay = true;
      audioEl.style.display = 'none';
      document.body.appendChild(audioEl);
    }
    return audioEl;
  };

  const detachTrack = (track) => {
    try {
      track.detach();
    } catch {
      /* already detached */
    }
    attachedTracks.delete(track);
    if (tracksByKind.get(track.kind) === track) {
      tracksByKind.delete(track.kind);
    }
  };

  const attachTrack = (track) => {
    if (!track || attachedTracks.has(track)) return;

    const previous = tracksByKind.get(track.kind);
    if (previous && previous !== track) detachTrack(previous);

    if (track.kind === Track.Kind.Video) {
      if (!videoElement) return;
      track.attach(videoElement);
      // Autoplay can be rejected before the first user gesture; retry quietly
      // instead of leaving a black <video> that "has" a track.
      videoElement.play?.().catch(() => {});
    } else if (track.kind === Track.Kind.Audio) {
      const el = getAudioEl();
      track.attach(el);
      el.play?.().catch(() => {});
    } else {
      return;
    }

    attachedTracks.add(track);
    tracksByKind.set(track.kind, track);
    resolveFirstTrack?.();
    if (!failed) onStatus?.('live');
  };

  room.on(RoomEvent.TrackSubscribed, (track) => {
    attachTrack(track);
  });

  room.on(RoomEvent.TrackUnsubscribed, (track) => {
    detachTrack(track);
    if (attachedTracks.size === 0 && !failed) {
      onStatus?.('done');
    }
  });

  // A transient signal drop must not look like a finished stream: report it
  // so the UI can show "Reconnecting…" and keep the last frame on screen.
  room.on(RoomEvent.Reconnecting, () => {
    if (!failed) onStatus?.('reconnecting');
  });
  room.on(RoomEvent.Reconnected, () => {
    if (!failed) onStatus?.(attachedTracks.size ? 'live' : 'connecting');
  });

  room.on(RoomEvent.ConnectionStateChanged, (state) => {
    if (state === 'disconnected' && !failed) {
      onStatus?.('done');
    }
  });

  room.on(RoomEvent.Disconnected, () => {
    if (!failed) onStatus?.('done');
  });

  try {
    await withTimeout(
      room.connect(livekitUrl, token),
      connectTimeoutMs,
      `no answer from ${livekitUrl} within ${Math.round(connectTimeoutMs / 1000)}s`,
      () => {
        try {
          room.disconnect();
        } catch {
          /* ignore */
        }
      }
    );

    // Check for tracks that were already published before/during connect
    for (const participant of room.remoteParticipants.values()) {
      for (const pub of participant.trackPublications.values()) {
        if (pub.track) {
          attachTrack(pub.track);
        }
      }
    }

    // Connected to the SFU but nobody published: the backend publisher either
    // failed or has not started. Wait a bounded time, then surface an error so
    // the caller can fall back instead of showing an empty player forever.
    if (attachedTracks.size === 0) {
      await withTimeout(
        firstTrack,
        trackTimeoutMs,
        `connected to ${livekitUrl} but the publisher sent no media within ${Math.round(trackTimeoutMs / 1000)}s`,
        () => {
          try {
            room.disconnect();
          } catch {
            /* ignore */
          }
        }
      );
    }
  } catch (err) {
    failed = true;
    onStatus?.('error');
    for (const track of Array.from(attachedTracks)) {
      detachTrack(track);
    }
    if (audioEl) {
      audioEl.remove();
      audioEl = null;
    }
    try {
      room.disconnect();
    } catch {
      /* ignore */
    }
    throw new Error(err?.message || String(err));
  }

  const disconnect = () => {
    // Intentional teardown: suppress the 'done'/'error' callbacks that the
    // SDK may still emit while disconnecting.
    failed = true;
    for (const track of Array.from(attachedTracks)) {
      detachTrack(track);
    }
    attachedTracks.clear();
    tracksByKind.clear();
    // Remove injected audio element
    if (audioEl) {
      audioEl.remove();
      audioEl = null;
    }
    try {
      room.disconnect();
    } catch {
      /* ignore */
    }
  };

  return { room, disconnect };
}
