/**
 * Shared backend telemetry poller.
 *
 * The header, telemetry panel, stage and live studio all need `/health`,
 * `/gpu` and `/livekit/status`. Polling each from its own timer produced
 * overlapping requests (one slow poll outliving a 6s interval) and up to 4
 * copies of the same call. This module owns a single, visibility-aware poller
 * and exposes a `useSyncExternalStore` hook so every consumer renders from the
 * same snapshot.
 *
 * Behaviour:
 * - one in-flight poll at a time (concurrent callers reuse the same promise)
 * - health + GPU in parallel, LiveKit status cached for `LIVEKIT_TTL_MS`
 * - offline only after `OFFLINE_AFTER_FAILURES` consecutive failures, with
 *   exponential backoff while the backend is down
 * - pauses in hidden tabs and refreshes immediately when the tab returns
 * - resets and re-polls when the API base changes
 */

import { useSyncExternalStore } from 'react';
import { fetchGpuMetrics, fetchHealth } from './api';
import { fetchLiveKitStatus } from './livekitHttp';

const POLL_MS = 6000;
const LIVEKIT_TTL_MS = 10000;
const MAX_BACKOFF_MS = 30000;
const OFFLINE_AFTER_FAILURES = 3;

const EMPTY = Object.freeze({
  health: null,
  gpu: null,
  liveKit: null,
  online: false,
  offline: false,
  checking: false,
  failures: 0,
  lastCheckedAt: null,
});

let snapshot = EMPTY;
let listeners = new Set();
let timer = null;
let inFlight = null;
let failures = 0;
let liveKitFetchedAt = 0;
let epoch = 0;
let started = false;

function emit(patch) {
  snapshot = { ...snapshot, ...patch };
  for (const listener of listeners) listener();
}

function getSnapshot() {
  return snapshot;
}

function nextDelay() {
  if (failures < OFFLINE_AFTER_FAILURES) return POLL_MS;
  return Math.min(POLL_MS * 2 ** (failures - (OFFLINE_AFTER_FAILURES - 1)), MAX_BACKOFF_MS);
}

function clearTimer() {
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
}

function schedule(delay) {
  clearTimer();
  if (!started) return;
  timer = setTimeout(() => {
    timer = null;
    poll()
      .catch(() => {})
      .finally(() => {
        if (started) schedule(nextDelay());
      });
  }, Math.max(0, delay));
}

/**
 * Run one poll. Resolves with the new snapshot; concurrent callers share the
 * same in-flight request.
 */
export function pollBackendStatus({ forceLiveKit = false } = {}) {
  if (inFlight) return inFlight;
  const pollEpoch = epoch;

  const task = (async () => {
    emit({ checking: true });

    const liveKitStale = forceLiveKit
      || !snapshot.liveKit
      || Date.now() - liveKitFetchedAt > LIVEKIT_TTL_MS;

    const [healthResult, gpuResult, liveKitResult] = await Promise.all([
      fetchHealth().then((value) => ({ ok: true, value }), (error) => ({ ok: false, error })),
      fetchGpuMetrics().then((value) => ({ ok: true, value }), (error) => ({ ok: false, error })),
      liveKitStale
        ? fetchLiveKitStatus().then((value) => ({ ok: true, value }), (error) => ({ ok: false, error }))
        : Promise.resolve(null),
    ]);

    // A manual refresh or API-base change happened while this poll was in
    // flight: discard the stale result rather than showing another host's data.
    if (pollEpoch !== epoch) return snapshot;

    const patch = {
      checking: false,
      lastCheckedAt: Date.now(),
    };

    if (healthResult.ok) {
      failures = 0;
      patch.health = healthResult.value;
      patch.online = true;
    } else {
      failures += 1;
      patch.online = failures < OFFLINE_AFTER_FAILURES ? snapshot.online : false;
      patch.health = snapshot.health;
    }
    // A GPU poll failure (no CUDA / endpoint absent) must not blank a widget
    // that already has data.
    if (gpuResult.ok) patch.gpu = gpuResult.value;

    if (liveKitResult) {
      liveKitFetchedAt = Date.now();
      patch.liveKit = liveKitResult.ok ? liveKitResult.value : { configured: false, reachable: false };
    }

    patch.failures = failures;
    patch.offline = failures >= OFFLINE_AFTER_FAILURES;
    emit(patch);
    return snapshot;
  })();

  let tracked;
  tracked = task.finally(() => {
    if (inFlight === tracked) inFlight = null;
  });
  inFlight = tracked;
  return tracked;
}

/** Manual refresh (Refresh button, after saving config, etc.). */
export function refreshBackendStatus({ forceLiveKit = true } = {}) {
  // A forced refresh should not reuse a poll that started before the change.
  if (forceLiveKit) {
    epoch += 1;
    inFlight = null;
    liveKitFetchedAt = 0;
  }
  return pollBackendStatus({ forceLiveKit });
}

function handleVisibility() {
  if (typeof document === 'undefined') return;
  if (document.visibilityState === 'hidden') {
    clearTimer();
    return;
  }
  pollBackendStatus().catch(() => {}).finally(() => schedule(nextDelay()));
}

function handleApiUrlChanged() {
  epoch += 1;
  inFlight = null;
  failures = 0;
  liveKitFetchedAt = 0;
  emit({
    health: null,
    gpu: null,
    liveKit: null,
    online: false,
    offline: false,
    checking: false,
    failures: 0,
    lastCheckedAt: null,
  });
  schedule(0);
}

function start() {
  if (started) return;
  started = true;
  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', handleVisibility);
  }
  if (typeof window !== 'undefined') {
    window.addEventListener('float:api_url_changed', handleApiUrlChanged);
  }
  schedule(0);
}

function stop() {
  started = false;
  clearTimer();
  if (typeof document !== 'undefined') {
    document.removeEventListener('visibilitychange', handleVisibility);
  }
  if (typeof window !== 'undefined') {
    window.removeEventListener('float:api_url_changed', handleApiUrlChanged);
  }
}

export function subscribeBackendStatus(listener) {
  listeners.add(listener);
  if (listeners.size === 1) start();
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) stop();
  };
}

/** Subscribe a component to the shared backend snapshot. */
export function useBackendStatus() {
  return useSyncExternalStore(subscribeBackendStatus, getSnapshot, getSnapshot);
}

/** Testing hook: stop timers/listeners and reset the snapshot. */
export function resetBackendStatusForTests() {
  stop();
  listeners = new Set();
  snapshot = EMPTY;
  inFlight = null;
  failures = 0;
  liveKitFetchedAt = 0;
  epoch = 0;
}
