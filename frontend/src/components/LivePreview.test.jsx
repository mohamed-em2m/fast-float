import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';

const sockets = [];

vi.mock('../services/live', () => ({
  openLiveSocket: vi.fn((requestId, handlers) => {
    const entry = { requestId, handlers, closed: false };
    sockets.push(entry);
    return {
      close: () => {
        entry.closed = true;
      },
    };
  }),
}));
vi.mock('../services/api', () => ({
  attachRemoteTrack: vi.fn(() => false),
  createWebRtcSession: vi.fn(async () => ({ close: vi.fn(), connectionState: 'new' })),
}));
vi.mock('../services/livekit', () => ({
  connectLiveKitSession: vi.fn(async () => ({ room: {}, disconnect: vi.fn() })),
}));
vi.mock('../services/livekitHttp', () => ({
  fetchLiveKitPublishStatus: vi.fn(async () => null),
}));

import LivePreview from './LivePreview';

function renderWs(props = {}) {
  return render(
    <LivePreview
      requestId="req-1"
      transport="ws"
      onStatus={() => {}}
      onFallback={() => {}}
      onEnded={() => {}}
      {...props}
    />
  );
}

beforeEach(() => {
  sockets.length = 0;
});

afterEach(() => {
  vi.useRealTimers();
});

describe('LivePreview over WebSocket', () => {
  it('treats a close after frames as a finished render', async () => {
    const onEnded = vi.fn();
    const onFallback = vi.fn();
    renderWs({ onEnded, onFallback });
    const { handlers } = sockets[0];

    act(() => {
      handlers.onMeta({ width: 4, height: 4, channels: 3 });
    });
    expect(screen.getByText(/Live/)).toBeInTheDocument();

    act(() => {
      handlers.onStatus('closed');
    });

    expect(onEnded).toHaveBeenCalledWith('req-1');
    expect(onFallback).not.toHaveBeenCalled();
    expect(screen.getByText(/Finished/)).toBeInTheDocument();
  });

  it('falls back when the socket closes before sending frames', () => {
    const onFallback = vi.fn();
    const onEnded = vi.fn();
    renderWs({ onFallback, onEnded });
    const { handlers } = sockets[0];

    act(() => {
      handlers.onStatus('closed');
    });

    expect(onFallback).toHaveBeenCalledWith(expect.stringMatching(/closed before sending frames/i));
    expect(onEnded).not.toHaveBeenCalled();
  });

  it('falls back when the handshake never answers', async () => {
    vi.useFakeTimers();
    const onFallback = vi.fn();
    renderWs({ onFallback });
    const entry = sockets[0];

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15000);
    });

    expect(onFallback).toHaveBeenCalledWith(expect.stringMatching(/did not answer/i));
    expect(entry.closed).toBe(true);
  });

  it('closes the socket on unmount', async () => {
    const { unmount } = renderWs();
    const entry = sockets[0];
    unmount();
    await waitFor(() => expect(entry.closed).toBe(true));
  });
});
