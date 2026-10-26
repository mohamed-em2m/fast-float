import { useEffect, useRef, useState } from 'react';
import { Play } from 'lucide-react';
import { attachRemoteTrack, createWebRtcSession } from '../services/api';
import { openLiveSocket } from '../services/live';
import { connectLiveKitSession } from '../services/livekit';
import { fetchLiveKitPublishStatus } from '../services/livekitHttp';

const STATUS_LABEL = {
  idle: 'Idle',
  connecting: 'Connecting…',
  reconnecting: 'Reconnecting…',
  live: 'Live',
  done: 'Finished',
  error: 'Error',
};

/**
 * In-flight preview of a running generation.
 *
 * transport === 'ws'      raw frames over WebSocket, drawn to a canvas
 * transport === 'webrtc'  the same frames re-encoded by aiortc into <video>
 * transport === 'livekit' frames streamed via LiveKit SFU into <video>
 *
 * When the LiveKit SFU cannot be reached (not configured, dead cloud project,
 * blocked WebRTC/UDP) the studio first falls back to the raw WebRTC transport
 * (signalled by this API host, ICE via its built-in STUN responder); only if
 * that also fails does it degrade to the WebSocket transport.
 */
export default function LivePreview({ requestId, transport, onStatus, onFallback, onEnded }) {
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const [status, setStatus] = useState('connecting');
  const [gaps, setGaps] = useState(0);
  const [size, setSize] = useState(null);
  const [needsGesture, setNeedsGesture] = useState(false);
  // Kept in a ref so a new callback identity never restarts the socket.
  const onStatusRef = useRef(onStatus);
  const onEndedRef = useRef(onEnded);

  useEffect(() => {
    onStatusRef.current = onStatus;
    onEndedRef.current = onEnded;
  });

  useEffect(() => {
    if (!requestId) return undefined;
    let cancelled = false;
    let ended = false;

    const report = (s) => {
      if (cancelled) return;
      setStatus(s);
      // A transport can report 'done' more than once (track unsubscribe +
      // room disconnect); let the parent switch to the MP4 exactly once.
      if (s === 'done' && !ended) {
        ended = true;
        onEndedRef.current?.(requestId);
      }
    };

    if (transport === 'ws') {
      let meta = null;
      let sawMeta = false;
      const WS_CONNECT_TIMEOUT_MS = 15000;
      let connectTimer = null;
      // Frame buffers are reused across frames: at 512x512 a fresh RGBA buffer
      // plus ImageData per frame allocated ~1 MB at 25 fps and kept the GC busy.
      let frameBuffer = null;
      let frameImage = null;
      const draw = ({ bytes }) => {
        const canvas = canvasRef.current;
        if (!canvas || !meta) return;
        const { width, height, channels } = meta;
        const ctx = canvas.getContext('2d');
        // A canvas without 2D support (headless/embedded webviews) must not
        // throw on every incoming frame.
        if (!ctx) return;
        const pixels = width * height;
        if (!frameBuffer || frameBuffer.length !== pixels * 4) {
          frameBuffer = new Uint8ClampedArray(pixels * 4);
          frameImage = new ImageData(frameBuffer, width, height);
        }
        if (channels === 4) {
          frameBuffer.set(bytes.subarray(0, pixels * 4));
        } else {
          for (let i = 0, j = 0, k = 0; i < pixels; i += 1, j += 3, k += 4) {
            frameBuffer[k] = bytes[j];
            frameBuffer[k + 1] = bytes[j + 1];
            frameBuffer[k + 2] = bytes[j + 2];
            frameBuffer[k + 3] = 255;
          }
        }
        ctx.putImageData(frameImage, 0, 0);
      };

      const failSocket = (reason) => {
        if (cancelled) return;
        report('error');
        if (onFallback) onFallback(reason);
        else onStatusRef.current?.(reason);
      };

      const socket = openLiveSocket(requestId, {
        onMeta: (m) => {
          meta = m;
          sawMeta = true;
          if (connectTimer) {
            clearTimeout(connectTimer);
            connectTimer = null;
          }
          // Size the bitmap immediately so the first frame is not drawn into a
          // stale canvas; the state keeps React's props in sync for re-renders.
          const canvas = canvasRef.current;
          if (canvas) {
            canvas.width = m.width;
            canvas.height = m.height;
          }
          setSize({ width: m.width, height: m.height });
          report('live');
        },
        onFrame: draw,
        onGap: () => setGaps((n) => n + 1),
        onDone: () => report('done'),
        onError: (message) => {
          // The POST /generate response surfaces the failure; keep this visual.
          console.warn('[live] stream error:', message);
          report('error');
        },
        onStatus: (s) => {
          if (s === 'error') {
            failSocket('the live frame socket failed');
          } else if (s === 'closed') {
            // A clean close after frames means the render ended; a close
            // before the first meta message is a failed handshake.
            if (sawMeta) report('done');
            else failSocket('the live frame socket closed before sending frames');
          }
        },
      });

      // A socket can stay in CONNECTING forever behind a proxy that never
      // upgrades. Bound the handshake so the stage can fall back to the MP4.
      connectTimer = setTimeout(() => {
        if (cancelled || sawMeta) return;
        failSocket('the live frame socket did not answer in 15s');
        socket.close();
      }, WS_CONNECT_TIMEOUT_MS);

      return () => {
        cancelled = true;
        if (connectTimer) clearTimeout(connectTimer);
        socket.close();
      };
    }

    if (transport === 'livekit') {
      let livekitSession = null;
      let fellBack = false;
      const videoEl = videoRef.current;

      // The SFU can fail in two places: the handshake (thrown) or after the
      // room connected but the publisher never sent tracks (onStatus 'error').
      // Both must degrade exactly once, with the server's publisher error when
      // there is one, so the toast says *why* LiveKit failed.
      const failOver = async (reason) => {
        if (cancelled || fellBack) return;
        fellBack = true;
        let detail = reason || 'LiveKit connection failed';
        try {
          const status = await fetchLiveKitPublishStatus(requestId);
          if (status?.error) detail = status.error;
        } catch {
          /* best effort */
        }
        if (cancelled) return;
        report('error');
        if (onFallback) onFallback(detail);
        else onStatusRef.current?.(detail);
      };

      connectLiveKitSession({
        room: requestId,
        videoElement: videoEl,
        onStatus: (s) => {
          if (s === 'error') {
            // SFU unreachable → try the raw WebRTC transport next.
            failOver('LiveKit connection failed');
            return;
          }
          report(s);
          if (s === 'live' && videoEl) {
            videoEl.play().catch(() => {
              if (!cancelled) setNeedsGesture(true);
            });
          }
        },
      })
        .then((session) => {
          if (cancelled || fellBack) {
            session.disconnect();
            return;
          }
          livekitSession = session;
        })
        .catch((err) => {
          console.warn('[livekit] connection failed:', err);
          failOver(err?.message || 'LiveKit connection failed');
        });

      return () => {
        cancelled = true;
        try {
          livekitSession?.disconnect();
        } catch {
          /* ignore */
        }
        try {
          videoEl?.srcObject?.getTracks().forEach((track) => track.stop());
        } catch {
          /* ignore */
        }
      };
    }

    // WebRTC: negotiation starts immediately; the server attaches the live
    // track (and the request audio) to the peer it creates for this offer.
    // SDP success is not enough: ICE can stay in "connecting" forever behind a
    // TCP-only proxy or strict NAT. Bound that wait and degrade to WebSocket.
    let pc = null;
    let connectionTimer = null;
    let fellBack = false;
    const videoEl = videoRef.current;
    const ICE_TIMEOUT_MS = 12000;
    // aiortc keeps the peer open when the live track finishes; the remote
    // MediaStreamTrack's 'ended' event is what tells us the clip is over.
    const watchedTracks = new Set();
    const trackCleanups = [];
    const watchTrackEnd = () => {
      const media = videoEl?.srcObject;
      for (const track of media?.getTracks?.() || []) {
        if (track.kind !== 'video' || watchedTracks.has(track)) continue;
        watchedTracks.add(track);
        const onEnded = () => report('done');
        track.addEventListener('ended', onEnded);
        trackCleanups.push(() => track.removeEventListener('ended', onEnded));
      }
    };
    // Some browsers never fire 'ended' for a remote track and the peer can
    // take seconds to report the close. Watch actual playback progress instead
    // of assuming requestVideoFrameCallback exists: Safari/Firefox fall back to
    // currentTime advancing, so a healthy stream is never mistaken for a stall.
    const VIDEO_STALL_MS = 6000;
    let lastProgressAt = performance.now();
    let lastMediaTime = -1;
    const markFrame = () => {
      lastProgressAt = performance.now();
      videoEl?.requestVideoFrameCallback?.(markFrame);
    };
    if (typeof videoEl?.requestVideoFrameCallback === 'function') {
      videoEl.requestVideoFrameCallback(markFrame);
    }
    const stallTimer = setInterval(() => {
      if (cancelled || fellBack || !videoEl) return;
      // A paused/buffering element is not a finished clip; restart the clock
      // so autoplay delays do not trip the watchdog.
      if (videoEl.paused || videoEl.readyState < 2) {
        lastProgressAt = performance.now();
        lastMediaTime = videoEl.currentTime;
        return;
      }
      const mediaTime = videoEl.currentTime;
      if (mediaTime !== lastMediaTime) {
        lastMediaTime = mediaTime;
        lastProgressAt = performance.now();
        return;
      }
      if (performance.now() - lastProgressAt > VIDEO_STALL_MS) report('done');
    }, 1000);

    const clearConnectionTimer = () => {
      if (connectionTimer) {
        clearTimeout(connectionTimer);
        connectionTimer = null;
      }
    };

    const failOver = (reason) => {
      if (cancelled || fellBack) return;
      fellBack = true;
      clearConnectionTimer();
      report('error');
      if (onFallback) onFallback(reason);
      else onStatusRef.current?.(reason);
    };

    const armConnectionTimer = () => {
      clearConnectionTimer();
      connectionTimer = setTimeout(() => {
        if (cancelled || fellBack) return;
        if (pc?.connectionState === 'connected') return;
        failOver('ICE timeout — no TURN relay or reachable UDP path');
      }, ICE_TIMEOUT_MS);
    };

    createWebRtcSession(
      requestId,
      (event) => {
        // Video and audio may arrive as separate streams; merge them so the
        // second ontrack does not replace the first track.
        if (attachRemoteTrack(videoEl, event)) {
          watchTrackEnd();
          clearConnectionTimer();
          videoEl.play().catch(() => {
            if (!cancelled) setNeedsGesture(true);
          });
          report('live');
        }
      },
      (state) => {
        if (state === 'connected') {
          clearConnectionTimer();
          report('live');
        } else if (state === 'failed') {
          failOver('ICE negotiation failed — no reachable media path');
        } else if (state === 'closed') {
          clearConnectionTimer();
          report('done');
        }
        // 'disconnected' is often transient (ICE restart); the stall watchdog
        // and the remote track's 'ended' event decide when it is really over.
      },
      { live: true }
    )
      .then((connection) => {
        if (cancelled) {
          connection.close();
          return;
        }
        pc = connection;
        if (connection.connectionState === 'connected') {
          clearConnectionTimer();
          report('live');
        } else {
          armConnectionTimer();
        }
      })
      .catch((err) => {
        // Negotiation itself failed (not just a dropped peer): let the parent
        // degrade to the WebSocket transport.
        failOver(err.message || 'WebRTC negotiation failed');
      });

    return () => {
      cancelled = true;
      clearInterval(stallTimer);
      clearConnectionTimer();
      for (const cleanup of trackCleanups) {
        try {
          cleanup();
        } catch {
          /* track already gone */
        }
      }
      try {
        pc?.close();
      } catch {
        /* already closed */
      }
      try {
        videoEl?.srcObject?.getTracks().forEach((track) => track.stop());
      } catch {
        /* ignore */
      }
    };
  }, [requestId, transport, onFallback]);

  return (
    <div className="live-preview">
      {transport === 'ws' ? (
        <canvas
          ref={canvasRef}
          className="live-preview-media"
          width={size?.width ?? 512}
          height={size?.height ?? 512}
        />
      ) : (
        <video
          ref={videoRef}
          className="live-preview-media"
          autoPlay
          playsInline
          controls
          onPlaying={() => setNeedsGesture(false)}
        />
      )}
      {needsGesture && transport !== 'ws' && (
        <button
          type="button"
          className="live-preview-play"
          onClick={async () => {
            try {
              await videoRef.current?.play();
              setNeedsGesture(false);
            } catch {
              /* the overlay stays until the browser allows playback */
            }
          }}
        >
          <Play size={17} fill="currentColor" />
          Tap to play with sound
        </button>
      )}
      <div className="live-preview-overlay">
        <span className={`live-preview-status live-preview-status--${status}`}>
          <span className="live-preview-dot" />
          {STATUS_LABEL[status] ?? status}
          {transport === 'ws' ? ' · WebSocket' : transport === 'livekit' ? ' · LiveKit' : ' · WebRTC'}
        </span>
        {gaps > 0 && <span className="live-preview-gaps">{gaps} dropped</span>}
      </div>
    </div>
  );
}
