import React from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';

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
// Stand in for the microphone: hand the app a ready clip immediately.
vi.mock('./components/AudioEngine', () => ({
  default: function AudioEngineStub({ setAudioBase64 }) {
    React.useEffect(() => {
      setAudioBase64('AAAA');
    }, [setAudioBase64]);
    return <div data-testid="audio-stub" />;
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
});
