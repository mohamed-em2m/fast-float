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

function ms(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '--';
  return `${Number(value).toFixed(digits)} ms`;
}

function clockTime(seconds) {
  if (!seconds) return '';
  return new Date(seconds * 1000).toLocaleTimeString();
}

/**
 * One metric: label (+ hint) on the left, the number in the shared,
 * right-aligned value column. Every row has the same height, so the two
 * halves of the panel keep their dividers on the same lines.
 */
function MetricRow({ icon: Icon, tone = 'text-cyan', label, value, hint }) {
  return (
    <div className="perf-row">
      <div className="perf-row-label">
        <Icon size={14} className={tone} />
        <div className="perf-row-text">
          <span className="perf-row-title">{label}</span>
          <span className="perf-row-hint">{hint}</span>
        </div>
      </div>
      <div className="perf-row-value">{value}</div>
    </div>
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
    ? `${num(points[points.length - 1].worst_stream_fps, 1)} @ ${points[points.length - 1].concurrency} streams`
    : '--';

  const requestCount = requests?.count || 0;
  const recent = requests?.recent || [];
  const dist = (key) => requests?.[key] || null;
  const distPair = (key, digits = 1, suffix = '') => {
    const d = dist(key);
    if (!d || !d.count) return '--';
    return `${num(d.p50, digits, suffix)} / ${num(d.p95, digits, suffix)}`;
  };
  const split = baseline
    ? `${num(baseline.model_seconds, 2)} / ${num(baseline.encode_seconds, 2)} / ${num(
        baseline.other_seconds,
        2
      )} s`
    : '--';

  return (
    <div className="telemetry-section glass-panel perf-panel">
      <div className="section-header">
        <div className="section-title-wrap">
          <Gauge size={18} className="section-icon text-cyan" />
          <h2 className="section-title">Performance Matrix</h2>
          {state?.finished_at ? (
            <span className="perf-subtitle-note">measured {clockTime(state.finished_at)}</span>
          ) : null}
        </div>
        <div className="perf-actions">
          <span className="perf-levels-label">streams</span>
          <input
            className="perf-levels-input"
            value={levels}
            onChange={(event) => setLevels(event.target.value)}
            placeholder="1,4,8,16"
            title="Stream counts to sweep"
            disabled={running}
          />
          <button
            className="btn-primary btn-sm perf-run"
            onClick={runBenchmark}
            disabled={running}
            title="Measure the matrix on this GPU"
          >
            {running ? <Loader2 size={14} className="spin" /> : <Play size={14} />}
            {running ? 'Measuring' : 'Measure'}
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
          <span>
            Benchmark running — measuring model-only fps, end-to-end fps and the concurrency sweep.
          </span>
        </div>
      )}

      {!matrix && !running && status !== 'error' && (
        <div className="perf-notice">
          <Signal size={14} />
          <span>
            No measurement yet. Press measure to benchmark this GPU (it holds the model while it runs).
          </span>
        </div>
      )}

      {/* Two mirrored halves: speed/throughput on the left, latency on the right.
          Both use the same row layout, so every number shares one column. */}
      <div className="perf-metrics">
        <section className="perf-metric-col">
          <h3 className="perf-subtitle">Speed and throughput</h3>
          <MetricRow
            icon={Activity}
            label="Throughput fps (model-only)"
            value={num(matrix?.model_fps, 2)}
            hint="ODE solve + decode, no encode"
          />
          <MetricRow
            icon={Layers}
            tone="text-indigo"
            label="End-to-end fps"
            value={num(matrix?.e2e_fps, 2)}
            hint="encode, composite, mux, write"
          />
          <MetricRow
            icon={Signal}
            label="Real-time factor"
            value={matrix ? `${num(matrix.rtf, 2)}x` : '--'}
            hint={`against a ${num(matrix?.target_fps ?? 25, 0)} fps target`}
          />
          <MetricRow
            icon={Gauge}
            tone="text-indigo"
            label="Streams at real-time"
            value={matrix ? matrix.streams_at_realtime : '--'}
            hint="largest level at the target fps"
          />
          <MetricRow
            icon={Timer}
            label="Per-stream fps under load"
            value={perStreamUnderLoad}
            hint="worst stream of the sweep"
          />
          <MetricRow
            icon={Layers}
            tone="text-indigo"
            label="Model / encode / other"
            value={split}
            hint="seconds of one single-stream run"
          />
        </section>

        <section className="perf-metric-col">
          <h3 className="perf-subtitle">Latency</h3>
          <MetricRow
            icon={Timer}
            label="Time to first frame (TTFF)"
            value={ms(baseline?.ttff_ms)}
            hint="trigger to first frame at transport"
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
            value={
              latency?.count
                ? `${num(latency.p50_ms, 1)} / ${num(latency.p95_ms, 1)} / ${num(latency.p99_ms, 1)}`
                : '--'
            }
            hint="ms per frame, vs its due time"
          />
          <MetricRow
            icon={Signal}
            tone="text-indigo"
            label="Frame-time jitter"
            value={ms(baseline?.jitter_ms, 2)}
            hint="stddev of the delivery intervals"
          />
          <MetricRow
            icon={AlertTriangle}
            tone="text-amber"
            label="Dropped frames"
            value={
              baseline
                ? `${baseline.dropped ?? 0} (${num((baseline.drop_ratio || 0) * 100, 2)}%)`
                : '--'
            }
            hint="overflow or too-late frames"
          />
        </section>
      </div>

      {points.length > 0 && (
        <section className="perf-block">
          <h3 className="perf-subtitle">
            Concurrency sweep
            <span className="perf-subtitle-note">streams sharing one GPU</span>
          </h3>
          <div className="perf-table-wrap">
            <table className="perf-data-table">
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
                    <td>{point.concurrency}</td>
                    <td>{num(point.aggregate_fps, 2)}</td>
                    <td>{num(point.per_stream_fps, 2)}</td>
                    <td>{num(point.worst_stream_fps, 2)}</td>
                    <td>{num(point.rtf, 2)}x</td>
                    <td>{num((point.gpu_saturation || 0) * 100, 1)}%</td>
                    <td>{point.dropped}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Rolling per-request view: what the box actually did, no synthetic load. */}
      <section className="perf-block">
        <h3 className="perf-subtitle">
          Recent requests
          <span className="perf-subtitle-note">
            {requestCount
              ? `${requestCount} in the last ${requests?.window || 200} · ${requests?.dropped_total || 0} dropped (${
                  requests?.drop_ratio ? num(requests.drop_ratio * 100, 3) : '0'
                }%)`
              : 'no requests yet'}
          </span>
        </h3>

        {requestCount > 0 ? (
          <>
            <div className="perf-metrics">
              <section className="perf-metric-col">
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
                  value={distPair('rtf', 2, 'x')}
                  hint="e2e fps / target fps"
                />
              </section>
              <section className="perf-metric-col">
                <MetricRow
                  icon={Timer}
                  tone="text-indigo"
                  label="Request time p50 / p95"
                  value={distPair('request_seconds', 2, ' s')}
                  hint="trigger to finished mp4"
                />
                <MetricRow
                  icon={Timer}
                  label="TTFF p50 / p95"
                  value={distPair('ttff_ms', 0, ' ms')}
                  hint="live streams only"
                />
                <MetricRow
                  icon={Signal}
                  tone="text-indigo"
                  label="Frame jitter p50 / p95"
                  value={distPair('jitter_ms', 2, ' ms')}
                  hint="stddev of the delivery intervals"
                />
              </section>
            </div>

            {recent.length > 0 && (
              <div className="perf-table-wrap">
                <table className="perf-data-table">
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
                          <td>{clockTime(sample.at)}</td>
                          <td>{sample.frames}</td>
                          <td>{num(sample.model_fps, 2)}</td>
                          <td>{num(sample.e2e_fps, 2)}</td>
                          <td>{num(sample.rtf, 2)}x</td>
                          <td>
                            {sample.ttff_ms === null || sample.ttff_ms === undefined
                              ? '--'
                              : ms(sample.ttff_ms, 0)}
                          </td>
                          <td>{sample.dropped}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        ) : (
          <div className="perf-notice">
            <Signal size={14} />
            <span>The window fills itself with every finished request — no benchmark needed.</span>
          </div>
        )}
      </section>

      {Array.isArray(matrix?.notes) && matrix.notes.length > 0 && (
        <details className="perf-notes-details">
          <summary>measurement notes</summary>
          <ul className="perf-notes">
            {matrix.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
