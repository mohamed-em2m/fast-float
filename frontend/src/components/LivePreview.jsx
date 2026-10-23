import { useEffect, useRef, useState } from 'react';
import { attachRemoteTrack, createWebRtcSession } from '../services/api';
import { openLiveSocket } from '../services/live';
import { connectLiveKitSession } from '../services/livekit';

const STATUS_LABEL = {
  idle: 'Idle',
  connecting: 'Connecting…',
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
export default function LivePreview({ requestId, transport, onStatus, onFallback }) {
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const [status, setStatus] = useState('connecting');
  const [gaps, setGaps] = useState(0);
  const [size, setSize] = useState(null);
  // Kept in a ref so a new callback identity never restarts the socket.
  const onStatusRef = useRef(onStatus);

  useEffect(() => {
    onStatusRef.current = onStatus;
  });

  useEffect(() => {
    if (!requestId) return undefined;
    let cancelled = false;

    const report = (s) => {
      if (!cancelled) setStatus(s);
    };

    if (transport === 'ws') {
      let meta = null;
      const draw = ({ bytes }) => {
        const canvas = canvasRef.current;
        if (!canvas || !meta) return;
        const { width, height, channels } = meta;
        const ctx = canvas.getContext('2d');
        if (channels === 4) {
          ctx.putImageData(new ImageData(new Uint8ClampedArray(bytes), width, height), 0, 0);
          return;
        }
        const pixels = width * height;
        const rgba = new Uint8ClampedArray(pixels * 4);
        for (let i = 0, j = 0; i < pixels; i += 1, j += 3) {
          rgba[i * 4] = bytes[j];
          rgba[i * 4 + 1] = bytes[j + 1];
          rgba[i * 4 + 2] = bytes[j + 2];
          rgba[i * 4 + 3] = 255;
        }
        ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
      };

      const socket = openLiveSocket(requestId, {
        onMeta: (m) => {
          meta = m;
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
          if (s === 'error') report('error');
        },
      });

      return () => {
        cancelled = true;
        socket.close();
      };
    }

    if (transport === 'livekit') {
      let livekitSession = null;
      const videoEl = videoRef.current;

      connectLiveKitSession({
        room: requestId,
        videoElement: videoEl,
        onStatus: (s) => {
          if (s === 'error') {
            // SFU unreachable → try the raw WebRTC transport next.
            if (onFallback) onFallback('LiveKit connection failed');
            else report('error');
            return;
          }
          report(s);
          if (s === 'live' && videoEl) {
            videoEl.play().catch(() => {});
          }
        },
      })
        .then((session) => {
          if (cancelled) {
            session.disconnect();
            return;
          }
          livekitSession = session;
        })
        .catch((err) => {
          console.warn('[livekit] connection failed:', err);
          if (onFallback) {
            onFallback(err.message || 'LiveKit connection failed');
            return;
          }
          report('error');
          onStatusRef.current?.(err.message);
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
          clearConnectionTimer();
          videoEl.play().catch(() => {});
          report('live');
        }
      },
      (state) => {
        if (state === 'connected') {
          clearConnectionTimer();
          report('live');
        } else if (state === 'failed') {
          failOver('ICE negotiation failed — no reachable media path');
        } else if (state === 'disconnected' || state === 'closed') {
          clearConnectionTimer();
          report('done');
        }
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
      clearConnectionTimer();
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
        <video ref={videoRef} className="live-preview-media" autoPlay playsInline controls />
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
