/**
 * LiveKit client transport for the live preview and the conversational avatar.
 *
 * Grounded in docs/LIVEKIT_INTEGRATION.md:
 * - joins a room using a short-lived token minted by POST /livekit/token
 * - subscribes to tracks published by the avatar worker (kind agent with lk.publish_on_behalf)
 * - attaches video + audio to the preview <video> element
 */

import { Room, RoomEvent, Track, createAudioAnalyser } from 'livekit-client';
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
  if (typeof window === 'undefined') return '';
  return localStorage.getItem(GEMINI_API_KEY_STORAGE_KEY) || '';
}

export function setGeminiApiKey(key) {
  if (typeof window === 'undefined') return;
  const clean = (key || '').trim();
  if (clean) {
    localStorage.setItem(GEMINI_API_KEY_STORAGE_KEY, clean);
  } else {
    localStorage.removeItem(GEMINI_API_KEY_STORAGE_KEY);
  }
}

export function getGeminiModel() {
  if (typeof window === 'undefined') return '';
  return localStorage.getItem(GEMINI_MODEL_STORAGE_KEY) || '';
}

export function setGeminiModel(model) {
  if (typeof window === 'undefined') return;
  const clean = (model || '').trim();
  if (clean) {
    localStorage.setItem(GEMINI_MODEL_STORAGE_KEY, clean);
  } else {
    localStorage.removeItem(GEMINI_MODEL_STORAGE_KEY);
  }
}

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
 *
 * `verify: true` dials the SFU once (mint a token + connect) and refreshes
 * the connectivity check, so the Config panel can prove credentials work
 * before a session is started.
 */
export async function fetchLiveKitStatus({ verify = false } = {}) {
  try {
    const suffix = verify ? '?verify=1' : '';
    const res = await fetch(`${getApiUrl()}/livekit/status${suffix}`);
    if (!res.ok) return { configured: false, reachable: true, warnings: ['LiveKit status request failed.'] };
    return { ...(await res.json()), reachable: true };
  } catch (err) {
    return {
      configured: false,
      reachable: false,
      warnings: [err?.message || 'Cannot reach the backend'],
    };
  }
}

/**
 * Mint a viewer token for a given room.
 *
 * `publishMicrophone` opts into a publish grant so the browser can send its
 * microphone into the room. The Gemini Live backend listens to that track
 * instead of receiving PCM over HTTP.
 */
export async function fetchLiveKitToken(room, name = 'FLOAT Viewer', { publishMicrophone = false } = {}) {
  const res = await fetch(`${getApiUrl()}/livekit/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ room, name, publish_mic: publishMicrophone }),
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
 * Start a live-audio session (agent + renderer + avatar worker).
 *
 * `agent` picks the conversation engine: `builtin` renders browser PCM in the
 * API process, `gemini` dispatches the external Gemini Live worker, which
 * listens to the room microphone and streams A/V back through the avatar.
 */
export async function startConversation({
  room = null,
  imageBase64 = null,
  imageFormat = 'jpg',
  agent = 'builtin',
  agentName = null,
  geminiApiKey = null,
  geminiModel = null,
} = {}) {
  const res = await fetch(`${getApiUrl()}/livekit/conversation/start`, {
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
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Could not start the live session (${res.status})`);
  }
  return await res.json();
}

/** Push one ~200 ms chunk of mono int16 PCM (16 kHz) to the avatar renderer. */
export async function pushConversationAudio(room, audioBase64, { flush = false } = {}) {
  const res = await fetch(`${getApiUrl()}/livekit/conversation/audio`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ room, audio_base64: audioBase64, sample_rate: 16000, flush }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Live audio rejected (${res.status})`);
  }
  return await res.json();
}

/** Barge-in: stop the avatar's current utterance immediately. */
export async function interruptConversation(room) {
  const res = await fetch(`${getApiUrl()}/livekit/conversation/interrupt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ room }),
  });
  if (!res.ok) return null;
  return await res.json().catch(() => null);
}

/** End the live session and stop its renderer/worker. */
export async function stopConversation(room) {
  const res = await fetch(`${getApiUrl()}/livekit/conversation/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ room }),
  });
  if (!res.ok) return null;
  return await res.json().catch(() => null);
}

/**
 * Connect to a LiveKit room and stream avatar tracks to the video element.
 *
 * Video *and* audio are attached to `videoElement`, so the browser renders
 * them from one MediaStream on a single clock. Attaching audio to a separate
 * hidden <audio> element gives it an independent jitter buffer and playback
 * start, which shows up as a constant lip-sync offset (and can be much worse
 * when that element's autoplay is delayed). A hidden element is only used as
 * a fallback when no video element is available.
 *
 * @param {Object} options
 * @param {string} options.room - Room name to join
 * @param {HTMLMediaElement} options.videoElement - <video> element to attach the video track to
 * @param {function} [options.onStatus] - Callback for status updates ('connecting', 'reconnecting', 'live', 'done', 'error')
 * @param {number} [options.connectTimeoutMs] - Bound on the SFU handshake
 * @param {number} [options.trackTimeoutMs] - Bound on waiting for the first track
 * @param {boolean} [options.publishMicrophone] - Publish the browser microphone (Gemini Live mode)
 * @returns {Promise<{ room: Room, disconnect: function }>}
 */
export async function connectLiveKitSession({
  room: roomName,
  videoElement,
  onStatus,
  connectTimeoutMs = CONNECT_TIMEOUT_MS,
  trackTimeoutMs = FIRST_TRACK_TIMEOUT_MS,
  publishMicrophone = false,
}) {
  onStatus?.('connecting');

  // 1. Get token
  const tokenData = await fetchLiveKitToken(roomName, 'FLOAT Viewer', { publishMicrophone });
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
  let reconnecting = false;
  let resolveFirstTrack;
  const firstTrack = new Promise((resolve) => {
    resolveFirstTrack = resolve;
  });

  // Fallback audio element, used only when no video element is available.
  // With a <video> we attach both tracks to it so they share one clock.
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
      if (videoElement) {
        // LiveKit merges the track into the element's existing MediaStream,
        // so video and audio share one playback clock.
        track.attach(videoElement);
        videoElement.play?.().catch(() => {});
      } else {
        const el = getAudioEl();
        track.attach(el);
        el.play?.().catch(() => {});
      }
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
    // A reconnect unsubscribes then resubscribes the publisher's tracks; that
    // is not the end of the stream, so do not flash "Finished" mid-session.
    if (attachedTracks.size === 0 && !failed && !reconnecting) {
      onStatus?.('done');
    }
  });

  // A transient signal drop must not look like a finished stream: report it
  // so the UI can show "Reconnecting…" and keep the last frame on screen.
  room.on(RoomEvent.Reconnecting, () => {
    reconnecting = true;
    if (!failed) onStatus?.('reconnecting');
  });
  room.on(RoomEvent.Reconnected, () => {
    reconnecting = false;
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

    // Gemini Live subscribes to the browser microphone, so the publisher has
    // to exist before the agent starts listening. Enabling it here keeps the
    // permission prompt attached to the click that started the session.
    if (publishMicrophone) {
      await room.localParticipant.setMicrophoneEnabled(true);
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

/** Mute or unmute the published microphone of a Gemini Live session. */
export async function setRoomMicrophoneMuted(room, muted) {
  if (!room?.localParticipant) return false;
  await room.localParticipant.setMicrophoneEnabled(!muted);
  return true;
}

/**
 * Sample the room microphone level for the studio meter.
 *
 * Only valid after `connectLiveKitSession({ publishMicrophone: true })`;
 * returns a no-op cleanup when no microphone track is published.
 */
export function createMicrophoneMeter(room, onLevel, { intervalMs = 120 } = {}) {
  const track = room?.localParticipant?.getTrackPublication?.(Track.Source.Microphone)?.track;
  if (!track) return () => {};
  const { calculateVolume, cleanup } = createAudioAnalyser(track, { smoothingTimeConstant: 0.3 });
  const timer = setInterval(() => onLevel?.(calculateVolume()), intervalMs);
  return () => {
    clearInterval(timer);
    Promise.resolve(cleanup()).catch(() => {});
  };
}
