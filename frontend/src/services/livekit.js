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
 * @param {function} [options.onStatus] - Callback for status updates ('connecting', 'live', 'done', 'error')
 * @returns {Promise<{ room: Room, disconnect: function }>}
 */
export async function connectLiveKitSession({
  room: roomName,
  videoElement,
  onStatus,
}) {
  onStatus?.('connecting');

  // 1. Get token
  const tokenData = await fetchLiveKitToken(roomName);
  const livekitUrl = alignLiveKitUrl(tokenData.url);
  const token = tokenData.token;

  // 2. Create LiveKit Room
  const room = new Room({
    adaptiveStream: true,
    dynacast: true,
  });

  const attachedTracks = new Set();

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

  const attachTrack = (track) => {
    if (attachedTracks.has(track)) return;

    if (track.kind === Track.Kind.Video && videoElement) {
      track.attach(videoElement);
      attachedTracks.add(track);
      onStatus?.('live');
    } else if (track.kind === Track.Kind.Audio) {
      track.attach(getAudioEl());
      attachedTracks.add(track);
      onStatus?.('live');
    }
  };

  room.on(RoomEvent.TrackSubscribed, (track) => {
    attachTrack(track);
  });

  room.on(RoomEvent.TrackUnsubscribed, (track) => {
    track.detach();
    attachedTracks.delete(track);
    if (attachedTracks.size === 0) {
      onStatus?.('done');
    }
  });

  room.on(RoomEvent.ConnectionStateChanged, (state) => {
    if (state === 'disconnected') {
      onStatus?.('done');
    }
  });

  room.on(RoomEvent.Disconnected, () => {
    onStatus?.('done');
  });

  try {
    await room.connect(livekitUrl, token);

    // Check for tracks that were already published before/during connect
    for (const participant of room.remoteParticipants.values()) {
      for (const pub of participant.trackPublications.values()) {
        if (pub.track) {
          attachTrack(pub.track);
        }
      }
    }
  } catch (err) {
    onStatus?.('error');
    throw err;
  }

  const disconnect = () => {
    for (const track of attachedTracks) {
      track.detach();
    }
    attachedTracks.clear();
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
