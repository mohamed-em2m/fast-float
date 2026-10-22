/**
 * Live frame transport over WebSocket.
 *
 * The backend pushes frames as they are decoded:
 *   text   {"type": "meta" | "gap" | "done" | "error", ...}
 *   binary [u32 frame_idx][u32 frame_count][u32 pts_ms][rgb bytes]
 *
 * Follows the same-origin `/api` proxy as api.js, so it works from LAN/ngrok
 * clients too.
 */

import { getApiUrl } from './api';

const HEADER_BYTES = 12;

/** Build ws(s)://.../ws/stream/<id> from the active HTTP API base. */
export function liveStreamUrl(requestId) {
  const base = new URL(getApiUrl(), window.location.origin);
  base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
  const prefix = base.pathname.replace(/\/$/, '');
  base.pathname = `${prefix}/ws/stream/${encodeURIComponent(requestId)}`;
  base.search = '';
  base.hash = '';
  return base.toString();
}

/**
 * Open a live frame socket. Returns { close() }.
 *
 * Callbacks: onMeta(meta), onFrame({ idx, count, ptsMs, bytes }), onGap(gap),
 * onDone(result), onError(message), onStatus(status).
 */
export function openLiveSocket(requestId, handlers = {}) {
  const { onMeta, onFrame, onGap, onDone, onError, onStatus } = handlers;
  const ws = new WebSocket(liveStreamUrl(requestId));
  ws.binaryType = 'arraybuffer';
  let closedByUs = false;

  ws.onopen = () => onStatus?.('open');

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === 'meta') onMeta?.(msg);
      else if (msg.type === 'gap') onGap?.(msg);
      else if (msg.type === 'done') onDone?.(msg.result);
      else if (msg.type === 'error') onError?.(msg.message);
      return;
    }

    const view = new DataView(event.data);
    if (view.byteLength < HEADER_BYTES) return;
    const idx = view.getUint32(0);
    const count = view.getUint32(4);
    const ptsMs = view.getUint32(8);
    const bytes = new Uint8Array(event.data, HEADER_BYTES);
    onFrame?.({ idx, count, ptsMs, bytes });
  };

  ws.onerror = () => onStatus?.('error');
  ws.onclose = () => {
    if (!closedByUs) onStatus?.('closed');
  };

  return {
    close() {
      closedByUs = true;
      try {
        ws.close();
      } catch {
        /* already closing */
      }
    },
  };
}
