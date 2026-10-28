/**
 * LiveKit SDK transport for the live preview and the conversational avatar.
 *
 * Grounded in docs/LIVEKIT_INTEGRATION.md:
 * - joins a room using a short-lived token minted by POST /livekit/token
 * - subscribes to tracks published by the avatar worker (kind agent with lk.publish_on_behalf)
 * - attaches video + audio to the preview <video> element
 *
 * HTTP-only helpers live in `livekitHttp.js` so status/config UI can import
 * them without pulling the (large) `livekit-client` SDK into its bundle.
 */

import { Room, RoomEvent, Track, createAudioAnalyser } from 'livekit-client';
import { alignLiveKitUrl, fetchLiveKitToken, getLiveKitUrlOverride } from './livekitHttp';

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
