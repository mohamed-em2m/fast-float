import React from 'react';
import { RotateCw, WifiOff } from 'lucide-react';
import { refreshBackendStatus, useBackendStatus } from '../services/telemetry';

/**
 * Persistent banner once the shared poller has failed several times in a row.
 * A single failed tick (slow upload, GPU spike) must not flash this.
 */
export default function OfflineBanner() {
  const { online, offline, checking } = useBackendStatus();
  if (online || !offline) return null;

  return (
    <div className="offline-banner" role="alert">
      <WifiOff size={16} className="offline-banner-icon" />
      <div className="offline-banner-text">
        <strong>Backend unreachable.</strong>
        <span>
          {' '}The Studio cannot reach the API. Check that the server is running,
          or open <strong>Server settings</strong> to point the Studio elsewhere.
        </span>
      </div>
      <button
        type="button"
        className="btn-secondary btn-sm"
        onClick={() => refreshBackendStatus().catch(() => {})}
        disabled={checking}
      >
        <RotateCw size={14} className={checking ? 'animate-spin' : ''} />
        {checking ? 'Retrying…' : 'Retry'}
      </button>
    </div>
  );
}
