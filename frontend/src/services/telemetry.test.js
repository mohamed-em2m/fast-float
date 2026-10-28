import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./api', () => ({
  fetchHealth: vi.fn(),
  fetchGpuMetrics: vi.fn(),
}));
vi.mock('./livekitHttp', () => ({
  fetchLiveKitStatus: vi.fn(),
}));

import { fetchGpuMetrics, fetchHealth } from './api';
import { fetchLiveKitStatus } from './livekitHttp';
import {
  pollBackendStatus,
  refreshBackendStatus,
  resetBackendStatusForTests,
  subscribeBackendStatus,
} from './telemetry';

beforeEach(() => {
  vi.clearAllMocks();
  resetBackendStatusForTests();
  fetchHealth.mockResolvedValue({ model_loaded: true });
  fetchGpuMetrics.mockResolvedValue({ info: { name: 'GPU' } });
  fetchLiveKitStatus.mockResolvedValue({ configured: true });
});

afterEach(() => {
  resetBackendStatusForTests();
});

describe('shared backend poller', () => {
  it('publishes health, GPU and LiveKit in one snapshot', async () => {
    const snapshot = await pollBackendStatus();

    expect(snapshot.online).toBe(true);
    expect(snapshot.health).toEqual({ model_loaded: true });
    expect(snapshot.gpu).toEqual({ info: { name: 'GPU' } });
    expect(snapshot.liveKit).toEqual({ configured: true });
    expect(snapshot.failures).toBe(0);
    expect(snapshot.checking).toBe(false);
    expect(snapshot.lastCheckedAt).toBeTypeOf('number');
  });

  it('reuses one in-flight request for concurrent callers', async () => {
    let resolveHealth;
    fetchHealth.mockReturnValue(new Promise((resolve) => {
      resolveHealth = resolve;
    }));

    const first = pollBackendStatus();
    const second = pollBackendStatus();
    expect(second).toBe(first);

    resolveHealth({});
    await first;
    expect(fetchHealth).toHaveBeenCalledTimes(1);
  });

  it('flips offline only after three consecutive failures', async () => {
    await pollBackendStatus();
    fetchHealth.mockRejectedValue(new Error('connection refused'));

    const first = await pollBackendStatus();
    const second = await pollBackendStatus();
    expect(first.offline).toBe(false);
    expect(second.offline).toBe(false);

    const third = await pollBackendStatus();
    expect(third.online).toBe(false);
    expect(third.offline).toBe(true);
    expect(third.failures).toBe(3);
  });

  it('keeps the GPU widget populated when only the GPU call fails', async () => {
    await pollBackendStatus();
    fetchGpuMetrics.mockRejectedValue(new Error('no CUDA'));
    const snapshot = await pollBackendStatus();
    expect(snapshot.gpu).toEqual({ info: { name: 'GPU' } });
    expect(snapshot.online).toBe(true);
  });

  it('caches /livekit/status for its TTL and refreshes on demand', async () => {
    await pollBackendStatus();
    await pollBackendStatus();
    expect(fetchLiveKitStatus).toHaveBeenCalledTimes(1);

    await refreshBackendStatus();
    expect(fetchLiveKitStatus).toHaveBeenCalledTimes(2);
  });

  it('reports an unreachable LiveKit status without losing health', async () => {
    fetchLiveKitStatus.mockResolvedValue({ configured: false, reachable: false });
    const snapshot = await pollBackendStatus();
    expect(snapshot.online).toBe(true);
    expect(snapshot.liveKit).toEqual({ configured: false, reachable: false });
  });

  it('auto-polls from a subscriber and keeps rescheduling', async () => {
    // Regression: the scheduled tick called an undefined `poll`, so the very
    // first timer threw and the chain never rescheduled — the Studio only ever
    // updated on a manual refresh.
    vi.useFakeTimers();
    const listener = vi.fn();
    const unsubscribe = subscribeBackendStatus(listener);
    try {
      await vi.advanceTimersByTimeAsync(0);
      expect(fetchHealth).toHaveBeenCalledTimes(1);
      expect(listener).toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(6000);
      expect(fetchHealth).toHaveBeenCalledTimes(2);
      expect(fetchLiveKitStatus).toHaveBeenCalledTimes(1); // cached for its TTL
    } finally {
      unsubscribe();
      vi.useRealTimers();
    }
  });
});
