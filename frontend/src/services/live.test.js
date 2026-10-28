import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

class FakeWebSocket {
  constructor(url) {
    FakeWebSocket.instances.push(this);
    this.url = url;
    this.binaryType = 'blob';
    this.sent = [];
    this.closed = false;
  }

  close() {
    this.closed = true;
    this.onclose?.({});
  }

  // Test helpers
  open() {
    this.onopen?.({});
  }

  message(data) {
    this.onmessage?.({ data });
  }

  failClose() {
    this.onclose?.({ code: 1006 });
  }
}

FakeWebSocket.instances = [];

let live;
let streams;

beforeEach(async () => {
  FakeWebSocket.instances = [];
  vi.stubGlobal('WebSocket', FakeWebSocket);
  vi.resetModules();
  live = await import('./live');
  streams = await import('./api');
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('liveStreamUrl', () => {
  it('maps the API base onto the ws(s) stream path', () => {
    const url = new URL(live.liveStreamUrl('abc-123'));
    expect(url.protocol).toBe('ws:');
    expect(url.pathname).toBe('/api/ws/stream/abc-123');
    expect(streams.getApiUrl()).toContain('/api');
  });
});

describe('openLiveSocket', () => {
  it('routes text control messages to their handlers', () => {
    const handlers = {
      onMeta: vi.fn(),
      onGap: vi.fn(),
      onDone: vi.fn(),
      onError: vi.fn(),
    };
    live.openLiveSocket('r1', handlers);
    const ws = FakeWebSocket.instances[0];

    ws.open();
    ws.message(JSON.stringify({ type: 'meta', width: 512, height: 512, channels: 3 }));
    ws.message(JSON.stringify({ type: 'gap', start: 3, end: 5 }));
    ws.message(JSON.stringify({ type: 'done', result: { request_id: 'r1' } }));
    ws.message(JSON.stringify({ type: 'error', message: 'gpu fell over' }));

    expect(handlers.onMeta).toHaveBeenCalledWith(expect.objectContaining({ width: 512 }));
    expect(handlers.onGap).toHaveBeenCalledTimes(1);
    expect(handlers.onDone).toHaveBeenCalledWith({ request_id: 'r1' });
    expect(handlers.onError).toHaveBeenCalledWith('gpu fell over');
  });

  it('decodes the binary frame header', () => {
    const onFrame = vi.fn();
    live.openLiveSocket('r2', { onFrame });
    const ws = FakeWebSocket.instances[0];

    const payload = new Uint8Array([1, 2, 3, 4, 5, 6]);
    const buffer = new ArrayBuffer(12 + payload.length);
    const view = new DataView(buffer);
    view.setUint32(0, 7); // frame index
    view.setUint32(4, 30); // frame count
    view.setUint32(8, 280); // pts ms
    new Uint8Array(buffer, 12).set(payload);
    ws.message(buffer);

    expect(onFrame).toHaveBeenCalledTimes(1);
    const frame = onFrame.mock.calls[0][0];
    expect(frame).toMatchObject({ idx: 7, count: 30, ptsMs: 280 });
    expect(Array.from(frame.bytes)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('reports an unexpected close but stays quiet on an intentional one', () => {
    const onStatus = vi.fn();
    const socket = live.openLiveSocket('r3', { onStatus });
    const ws = FakeWebSocket.instances[0];

    ws.failClose();
    expect(onStatus).toHaveBeenCalledWith('closed');

    onStatus.mockClear();
    socket.close();
    expect(onStatus).not.toHaveBeenCalled();
    expect(ws.closed).toBe(true);
  });

  it('ignores malformed and truncated messages', () => {
    const handlers = { onMeta: vi.fn(), onFrame: vi.fn(), onError: vi.fn() };
    live.openLiveSocket('r4', handlers);
    const ws = FakeWebSocket.instances[0];

    ws.message('{not json');
    ws.message(new ArrayBuffer(4));

    expect(handlers.onMeta).not.toHaveBeenCalled();
    expect(handlers.onFrame).not.toHaveBeenCalled();
    expect(handlers.onError).not.toHaveBeenCalled();
  });
});
