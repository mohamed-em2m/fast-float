import React, { useState, useEffect } from 'react';
import {
  Activity,
  Gauge,
  Timer,
  Layers,
  Play,
  Loader2,
  AlertTriangle,
  Signal,
} from 'lucide-react';
import { fetchPerfMatrix, fetchPerfRequests, startPerfBenchmark } from '../services/api';

const POLL_MS = 3000;

/** Format a number with a fixed precision, or a dash when it is missing. */
function num(value, digits = 1, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(value)) return '--';
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function ms(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '--';
  return `${Number(value).toFixed(1)} ms`;
}

function MetricRow({ icon: Icon, tone = 'text-cyan', label, value, hint }) {
  return (
    <tr>
      <td className="perf-label">
        <Icon size={14} className={tone} />
        <span>{label}</span>
      </td>
      <td className="perf-value">{value}</td>
      <td className="perf-hint">{hint}</td>
    </tr>
  );
}

export default function PerfMatrixPanel({ onToast }) {
  const [state, setState] = useState(null);
  const [requests, setRequests] = useState(null);
  const [busy, setBusy] = useState(false);
  const [levels, setLevels] = useState('1,4,8,16');

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      const [matrix, recent] = await Promise.all([fetchPerfMatrix(), fetchPerfRequests()]);
      if (cancelled) return;
      if (matrix) setState(matrix);
      if (recent) setRequests(recent);
    };
    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const runBenchmark = async () => {
    const concurrency = levels
      .split(',')
      .map((value) => parseInt(value, 10))
      .filter((value) => Number.isFinite(value) && value > 0);
    if (!concurrency.length) {
      onToast?.('Enter at least one stream count, e.g. 1,4,8,16.', 'error');
      return;
    }
    setBusy(true);
    try {
      const started = await startPerfBenchmark({ concurrency });
      setState(started);
      onToast?.(
        'Benchmark running — it holds the GPU, so live requests queue until it finishes.',
        'info'
      );
    } catch (err) {
      onToast?.(err.message || 'Benchmark could not be started.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const status = state?.status || 'idle';
  const matrix = state?.matrix || null;
  const baseline = matrix?.baseline || null;
  const latency = baseline?.latency || null;
  const points = matrix?.points || [];
  const running = status === 'running' || busy;

  const perStreamUnderLoad = points.length
    ? points.map((point) => `${point.concurrency}: ${num(point.per_stream_fps, 1)}`).join('  ')
    : '--';

  const requestCount = requests?.count || 0;
  const recent = requests?.recent || [];
  const dist = (key) => requests?.[key] || null;
  const distPair = (key, digits = 1) => {
    const d = dist(key);
    if (!d || !d.count) return '--';
    return `${num(d.p50, digits)} / ${num(d.p95, digits)}`;
  };

  return (
    <div className="telemetry-section glass-panel perf-matrix-section">
      <div className="section-header">
        <div className="section-title-wrap">
          <Gauge size={18} className="section-icon text-cyan" />
          <h2 className="section-title">Performance Matrix</h2>
        </div>
        <div className="perf-actions">
          <input
            className="perf-levels-input"
            value={levels}
            onChange={(event) => setLevels(event.target.value)}
            placeholder="1,4,8,16"
            title="Stream counts to sweep"
            disabled={running}
          />
          <button
            className="btn-icon-subtle"
            onClick={runBenchmark}
            disabled={running}
            title="Measure the matrix on this GPU"
          >
            {running ? <Loader2 size={15} className="spin" /> : <Play size={15} />}
          </button>
        </div>
      </div>

      {status === 'error' && (
        <div className="perf-notice perf-notice-error">
          <AlertTriangle size={14} />
          <span>{state?.error || 'The benchmark failed.'}</span>
        </div>
      )}

      {running && (
        <div className="perf-notice">
          <Loader2 size={14} className="spin" />
          <span>Benchmark running — measuring model-only fps, end-to-end fps and the concurrency sweep.</span>
        </div>
      )}

      {!matrix && !running && status !== 'error' && (
        <div className="perf-notice">
          <Signal size={14} />
          <span>No measurement yet. Press play to measure this GPU (it holds the model while it runs).</span>
        </div>
      )}

      {matrix && (
        <>
          <table className="perf-table">
            <tbody>
              <MetricRow
                icon={Activity}
                label="Throughput fps (model-only)"
                value={num(matrix.model_fps, 2)}
                hint="ODE solve + encoders + decode, no encode"
              />
              <MetricRow
                icon={Layers}
                tone="text-indigo"
                label="End-to-end fps"
                value={num(matrix.e2e_fps, 2)}
                hint="encode, composite, mux and write included"
              />
              <MetricRow
                icon={Signal}
                label="Real-time factor"
                value={`${num(matrix.rtf, 2)}x`}
                hint={`against a ${num(matrix.target_fps, 0)} fps target`}
              />
              <MetricRow
                icon={Gauge}
                tone="text-indigo"
                label="Streams per GPU at real-time"
                value={matrix.streams_at_realtime ?? '--'}
                hint="largest sweep level that still holds the target"
              />
              <MetricRow
                icon={Timer}
                label="Per-stream fps under load"
                value={perStreamUnderLoad}
                hint="streams: fps"
              />
            </tbody>
          </table>

          <table className="perf-table">
            <tbody>
              <MetricRow
                icon={Timer}
                label="Time to first frame (TTFF)"
                value={ms(baseline?.ttff_ms)}
                hint="trigger to the first frame at the transport"
              />
              <MetricRow
                icon={Timer}
                tone="text-indigo"
                label="Audio-in to video-out"
                value={ms(baseline?.audio_in_to_video_out_ms)}
                hint="offline mode: equals TTFF"
              />
              <MetricRow
                icon={Timer}
                tone="text-indigo"
                label="Conversational latency"
                value={ms(baseline?.conversational_latency_ms)}
                hint="stop speaking to avatar moving"
              />
              <MetricRow
                icon={Activity}
                label="p50 / p95 / p99 latency"
                value={`${num(latency?.p50_ms, 1)} / ${num(latency?.p95_ms, 1)} / ${num(latency?.p99_ms, 1)} ms`}
                hint="per frame, vs its presentation time"
              />
              <MetricRow
                icon={Signal}
                tone="text-indigo"
                label="Frame-time jitter"
                value={ms(baseline?.jitter_ms)}
                hint="stddev of the delivery intervals"
              />
              <MetricRow
                icon={AlertTriangle}
                tone="text-amber"
                label="Dropped frames"
                value={`${baseline?.dropped ?? 0} (${num((baseline?.drop_ratio || 0) * 100, 2)}%)`}
                hint="overflow or too-late frames"
              />
            </tbody>
          </table>

          {points.length > 0 && (
            <div className="perf-sweep">
              <h3 className="perf-subtitle">Concurrency sweep</h3>
              <table className="perf-table perf-sweep-table">
                <thead>
                  <tr>
                    <th>streams</th>
                    <th>aggregate fps</th>
                    <th>per-stream fps</th>
                    <th>worst stream</th>
                    <th>rtf</th>
                    <th>model share</th>
                    <th>dropped</th>
                  </tr>
                </thead>
                <tbody>
                  {points.map((point) => (
                    <tr key={point.concurrency}>
                      <td className="perf-value">{point.concurrency}</td>
                      <td className="perf-value">{num(point.aggregate_fps, 2)}</td>
                      <td className="perf-value">{num(point.per_stream_fps, 2)}</td>
                      <td className="perf-value">{num(point.worst_stream_fps, 2)}</td>
                      <td className="perf-value">{num(point.rtf, 2)}x</td>
                      <td className="perf-value">{num(point.gpu_saturation * 100, 1)}%</td>
                      <td className="perf-value">{point.dropped}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {Array.isArray(matrix.notes) && matrix.notes.length > 0 && (
            <ul className="perf-notes">
              {matrix.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </>
      )}

      {/* Rolling per-request view: what the box actually did, no synthetic load. */}
      <div className="perf-sweep">
        <h3 className="perf-subtitle">
          Recent requests
          <span className="perf-subtitle-note">
            {requestCount ? `${requestCount} in the last ${requests?.window || 200}` : 'no requests yet'}
          </span>
        </h3>
        {requestCount > 0 ? (
          <>
            <table className="perf-table">
              <tbody>
                <MetricRow
                  icon={Activity}
                  label="Model fps p50 / p95"
                  value={distPair('model_fps', 2)}
                  hint="across the window"
                />
                <MetricRow
                  icon={Layers}
                  tone="text-indigo"
                  label="End-to-end fps p50 / p95"
                  value={distPair('e2e_fps', 2)}
                  hint="encode and write included"
                />
                <MetricRow
                  icon={Signal}
                  label="Real-time factor p50 / p95"
                  value={distPair('rtf', 2)}
                  hint="e2e fps / target fps"
                />
                <MetricRow
                  icon={Timer}
                  tone="text-indigo"
                  label="Request time p50 / p95"
                  value={`${num(dist('request_seconds')?.p50, 2)} / ${num(dist('request_seconds')?.p95, 2)} s`}
                  hint="trigger to finished mp4"
                />
                <MetricRow
                  icon={Timer}
                  label="TTFF p50 / p95"
                  value={distPair('ttff_ms', 0)}
                  hint="live streams only"
                />
                <MetricRow
                  icon={Signal}
                  tone="text-indigo"
                  label="Frame jitter p50 / p95"
                  value={distPair('jitter_ms', 2)}
                  hint="stddev of the delivery intervals"
                />
                <MetricRow
                  icon={AlertTriangle}
                  tone="text-amber"
                  label="Dropped frames"
                  value={`${requests?.dropped_total ?? 0} (${num((requests?.drop_ratio || 0) * 100, 3)}%)`}
                  hint="over the window"
                />
              </tbody>
            </table>

            {recent.length > 0 && (
              <table className="perf-table perf-sweep-table">
                <thead>
                  <tr>
                    <th>finished</th>
                    <th>frames</th>
                    <th>model fps</th>
                    <th>e2e fps</th>
                    <th>rtf</th>
                    <th>ttff</th>
                    <th>dropped</th>
                  </tr>
                </thead>
                <tbody>
                  {recent
                    .slice()
                    .reverse()
                    .map((sample) => (
                      <tr key={sample.at}>
                        <td className="perf-value">{new Date(sample.at * 1000).toLocaleTimeString()}</td>
                        <td className="perf-value">{sample.frames}</td>
                        <td className="perf-value">{num(sample.model_fps, 2)}</td>
                        <td className="perf-value">{num(sample.e2e_fps, 2)}</td>
                        <td className="perf-value">{num(sample.rtf, 2)}x</td>
                        <td className="perf-value">
                          {sample.ttff_ms === null || sample.ttff_ms === undefined
                            ? '--'
                            : `${num(sample.ttff_ms, 0)} ms`}
                        </td>
                        <td className="perf-value">{sample.dropped}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            )}
          </>
        ) : (
          <div className="perf-notice">
            <Signal size={14} />
            <span>
              The window fills itself with every finished request — no benchmark needed.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
