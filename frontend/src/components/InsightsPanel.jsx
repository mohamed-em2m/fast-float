import React, { useState } from 'react';
import { BarChart3, ChevronDown, Gauge, History } from 'lucide-react';
import TelemetryDashboard from './TelemetryDashboard';
import PerfMatrixPanel from './PerfMatrixPanel';

const TABS = [
  { id: 'session', label: 'Session', icon: History },
  { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3 },
];

/**
 * Secondary analytics, collapsed by default so the creative flow stays the
 * loudest thing on the page. Monitoring is one click away, not a wall.
 */
export default function InsightsPanel({ latestResult, history, onToast }) {
  const [tab, setTab] = useState('session');
  const [open, setOpen] = useState(false);

  return (
    <section className={`insights glass-panel ${open ? 'is-open' : ''}`}>
      <button
        type="button"
        className="insights-toggle"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <span className="insights-title">
          <Gauge size={16} className="text-cyan" />
          Performance &amp; history
        </span>
        <span className="insights-meta">
          <span className="insights-meta-text">
            {latestResult?.performance?.fps
              ? `${latestResult.performance.fps} fps last run`
              : 'GPU telemetry, latency and benchmarks'}
          </span>
          <ChevronDown size={16} className="insights-chevron" />
        </span>
      </button>

      {open && (
        <div className="insights-body">
          <div className="insights-tabs" role="tablist" aria-label="Insight views">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={tab === id}
                className={`insights-tab ${tab === id ? 'is-active' : ''}`}
                onClick={() => setTab(id)}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </div>

          {tab === 'session' ? (
            <TelemetryDashboard latestResult={latestResult} history={history} embedded />
          ) : (
            <PerfMatrixPanel onToast={onToast} embedded />
          )}
        </div>
      )}
    </section>
  );
}
