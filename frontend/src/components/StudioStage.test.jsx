import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';

vi.mock('../services/api', () => ({
  fetchProgress: vi.fn(),
  getApiUrl: () => 'http://api.test/api',
  setIceServers: vi.fn(),
}));
vi.mock('../services/telemetry', () => ({
  useBackendStatus: () => ({ liveKit: { configured: false, reachable: true } }),
}));
vi.mock('../services/clipboard', () => ({
  copyText: vi.fn(async () => true),
}));

import { fetchProgress } from '../services/api';
import { copyText } from '../services/clipboard';
import StudioStage from './StudioStage';

const RESULT = {
  request_id: 'req-1',
  stream_url: 'http://api.test/api/stream/req-1',
  download_url: 'http://api.test/api/download/req-1',
  download_backup: 'http://backup.test/download/req-1',
  performance: { fps: 25, inference_time: 3.2, total_time: 9.1 },
  file_size_mb: 1.4,
};

function renderStage(props = {}) {
  return render(
    <StudioStage
      onGenerate={() => {}}
      isGenerating={false}
      hasAudio
      onToast={() => {}}
      transport="mp4"
      generationResult={null}
      {...props}
    />
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  fetchProgress.mockResolvedValue(null);
});

describe('StudioStage progress', () => {
  it('shows a determinate bar for a chunked render', async () => {
    fetchProgress.mockResolvedValue({
      status: 'processing',
      stage: 'generating',
      percent: 45,
      completed_chunks: 2,
      total_chunks: 4,
      audio_seconds: 12.5,
      started_at: Date.now() / 1000 - 30,
      elapsed_s: 30,
    });

    renderStage({ isGenerating: true, progressRequestId: 'req-1' });

    const bar = await screen.findByRole('progressbar', { name: /generation progress/i });
    await waitFor(() => expect(bar).toHaveAttribute('aria-valuenow', '45'));
    expect(screen.getByText(/2\/4 chunks/)).toBeInTheDocument();
    expect(screen.getByText(/12\.5s clip/)).toBeInTheDocument();
  });

  it('never shows a frozen 5% bar for a single-pass render', async () => {
    fetchProgress.mockResolvedValue({
      status: 'processing',
      stage: 'generating',
      percent: 5,
      completed_chunks: 0,
      total_chunks: 1,
      audio_seconds: 15.04,
      started_at: Date.now() / 1000 - 65,
      elapsed_s: 65,
    });

    renderStage({ isGenerating: true, progressRequestId: 'req-1' });

    const bar = await screen.findByRole('progressbar', { name: /generation progress/i });
    await waitFor(() => expect(bar).toHaveClass('is-indeterminate'));
    expect(bar).not.toHaveAttribute('aria-valuenow');
    // The stage label comes from the backend, not a client-side guess.
    expect(screen.getByText('Solving the flow ODE')).toBeInTheDocument();
    expect(screen.getByText(/15\.0s clip/)).toBeInTheDocument();
    expect(screen.getByText('1:05')).toBeInTheDocument();
  });

  it('stops polling once the render finishes', async () => {
    fetchProgress.mockResolvedValue({ status: 'processing', percent: 40, total_chunks: 2 });
    const { rerender } = renderStage({ isGenerating: true, progressRequestId: 'req-1' });
    await waitFor(() => expect(fetchProgress).toHaveBeenCalled());

    rerender(
      <StudioStage
        onGenerate={() => {}}
        isGenerating={false}
        progressRequestId={null}
        hasAudio
        onToast={() => {}}
        transport="mp4"
      />
    );
    const callsAfterFinish = fetchProgress.mock.calls.length;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    expect(fetchProgress).toHaveBeenCalledTimes(callsAfterFinish);
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  });
});

describe('StudioStage output', () => {
  it('hides the previous take while a new render is in flight', () => {
    const { container } = renderStage({ isGenerating: true, generationResult: RESULT });
    expect(container.querySelector('video')).toBeNull();
    expect(screen.queryByRole('button', { name: /download mp4/i })).not.toBeInTheDocument();
  });

  it('shows the result and its stats when idle', () => {
    renderStage({ generationResult: RESULT });
    expect(screen.getByText('25 FPS')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /download mp4/i })).toBeInTheDocument();
  });

  it('falls back to download_backup when the primary URL fails', async () => {
    const fetchMock = vi.fn(async (url) => {
      if (String(url).includes('api.test')) return { ok: false, status: 502, statusText: 'Bad Gateway' };
      return {
        ok: true,
        blob: async () => new Blob(['video'], { type: 'video/mp4' }),
      };
    });
    vi.stubGlobal('fetch', fetchMock);
    const createObjectURL = vi.fn(() => 'blob:fake');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL, revokeObjectURL }));
    // jsdom cannot navigate; record the synthetic click instead.
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const onToast = vi.fn();

    const { container } = renderStage({ generationResult: RESULT, onToast });
    await act(async () => {
      screen.getByRole('button', { name: /download mp4/i }).click();
    });

    await waitFor(() => expect(onToast).toHaveBeenCalledWith('Download started.', 'success'));
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[1][0])).toContain('backup.test');
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(anchorClick).toHaveBeenCalledTimes(1);
    expect(anchorClick.mock.instances[0].download).toBe('float_talking_portrait_req-1.mp4');
    // The anchor is removed from the DOM after the click.
    expect(container.querySelector('a[download]')).toBeNull();
    anchorClick.mockRestore();
    vi.unstubAllGlobals();
  });

  it('reports a clipboard failure instead of claiming success', async () => {
    copyText.mockResolvedValueOnce(false);
    const onToast = vi.fn();

    renderStage({ generationResult: RESULT, onToast });
    await act(async () => {
      screen.getByRole('button', { name: /copy download link/i }).click();
    });

    expect(copyText).toHaveBeenCalledWith(RESULT.download_url);
    expect(onToast).toHaveBeenCalledWith(expect.stringMatching(/could not copy/i), 'error');
  });

  it('surfaces a failed render inline with a retry action', () => {
    const onRetry = vi.fn();
    renderStage({ generationError: 'CUDA out of memory', onRetry });

    expect(screen.getByRole('alert')).toHaveTextContent('CUDA out of memory');
    screen.getByRole('button', { name: /retry/i }).click();
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
