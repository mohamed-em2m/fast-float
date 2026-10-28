import React from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';

vi.mock('./services/api', () => ({
  generateVideo: vi.fn(),
  fetchProgress: vi.fn(async () => null),
  blobToBase64: vi.fn(async () => 'AAAA'),
  alignUrlToApi: (url) => url,
  getApiUrl: () => 'http://api.test/api',
  setIceServers: vi.fn(),
}));
vi.mock('./services/telemetry', () => ({
  useBackendStatus: () => ({
    health: null,
    gpu: null,
    liveKit: { configured: false, reachable: true },
    online: true,
    offline: false,
    checking: false,
  }),
  refreshBackendStatus: vi.fn(async () => {}),
}));
// The header owns settings/telemetry UI; this test is about the workspace.
vi.mock('./components/Header', () => ({
  default: () => <header data-testid="header-stub" />,
}));
vi.mock('./components/OfflineBanner', () => ({
  default: () => null,
}));
// The live transports pull in the LiveKit SDK and WebRTC; this test only cares
// about the POST/result lifecycle, so keep the stage on a lightweight stub.
vi.mock('./components/LivePreview', () => ({
  default: () => <div data-testid="live-preview-stub" />,
}));
// Stand in for the microphone: hand the app a ready clip immediately.
vi.mock('./components/AudioEngine', () => ({
  default: function AudioEngineStub({ setAudioBase64 }) {
    React.useEffect(() => {
      setAudioBase64('AAAA');
    }, [setAudioBase64]);
    return (
      <div data-testid="audio-stub">
        <button type="button" onClick={() => setAudioBase64('BBBB')}>
          Replace audio
        </button>
      </div>
    );
  },
}));

import { generateVideo } from './services/api';
import App from './App';

beforeEach(() => {
  vi.clearAllMocks();
  window.history.replaceState({}, '', '/');
  window.localStorage.clear();
});

describe('App generation flow', () => {
  it('keeps the workspace usable and shows a retryable error when a render fails', async () => {
    generateVideo.mockRejectedValue(new Error('Server busy — try again shortly'));

    render(<App />);

    const button = await screen.findByRole('button', { name: /generate video/i });
    await waitFor(() => expect(button).toBeEnabled());

    await act(async () => {
      button.click();
    });

    const alerts = await screen.findAllByRole('alert');
    expect(alerts.some((node) => /server busy/i.test(node.textContent))).toBe(true);
    expect(await screen.findByRole('button', { name: /retry/i })).toBeInTheDocument();
    // The stage returns to idle instead of leaving a spinner behind.
    expect(await screen.findByRole('button', { name: /generate video/i })).toBeEnabled();
    expect(screen.queryByRole('progressbar', { name: /generation progress/i })).toBeNull();
  });

  it('publishes the result and offers it for download', async () => {
    generateVideo.mockResolvedValue({
      request_id: 'req-9',
      stream_url: 'http://api.test/api/stream/req-9',
      download_url: 'http://api.test/api/download/req-9',
      performance: { fps: 25, inference_time: 2, total_time: 6 },
      file_size_mb: 1,
    });

    render(<App />);
    const button = await screen.findByRole('button', { name: /generate video/i });
    await waitFor(() => expect(button).toBeEnabled());

    await act(async () => {
      button.click();
    });

    expect(await screen.findByRole('button', { name: /download mp4/i })).toBeInTheDocument();
    expect(await screen.findByRole('button', { name: /generate another/i })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('dismisses a failed render once the voice clip changes', async () => {
    generateVideo.mockRejectedValue(new Error('Server busy — try again shortly'));

    render(<App />);
    const button = await screen.findByRole('button', { name: /generate video/i });
    await waitFor(() => expect(button).toBeEnabled());

    await act(async () => {
      button.click();
    });

    expect(await screen.findByText('Generation failed')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /replace audio/i }));

    await waitFor(() => expect(screen.queryByText('Generation failed')).not.toBeInTheDocument());
    // The Generate button remains usable for the new clip.
    expect(screen.getByRole('button', { name: /generate video/i })).toBeEnabled();
  });
});
