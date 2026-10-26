import React, { useEffect, useRef, useState } from 'react';
import {
  AudioLines,
  CircleStop,
  Mic,
  MicOff,
  Play,
  Radio,
  ShieldCheck,
  Sparkles,
  Square,
  TriangleAlert,
  Wand2,
  Zap,
} from 'lucide-react';
import { MicStreamer } from '../services/audio';
import { getApiUrl } from '../services/api';
import { connectLiveKitSession, createMicrophoneMeter, setRoomMicrophoneMuted } from '../services/livekit';
import {
  fetchConversationStatus,
  getGeminiApiKey,
  getGeminiModel,
  interruptConversation,
  startConversation,
  stopConversation,
} from '../services/livekitHttp';
import { getStored, setStored } from '../services/storage';
import { useBackendStatus } from '../services/telemetry';

const PHASE_LABEL = {
  idle: 'Ready when you are',
  starting: 'Warming up the avatar…',
  live: 'Live',
  stopping: 'Ending session…',
  error: 'Session failed',
};

const LEVEL_BARS = 28;

const ENGINE_STORAGE_KEY = 'float_live_engine';

/**
 * Conversation engines exposed by the API.
 *
 * `builtin` streams browser PCM into the in-process agent/renderer.
 * `gemini` publishes the microphone to LiveKit and lets the external
 * livekit-agents Gemini Live worker drive the avatar instead.
 */
const ENGINES = {
  builtin: {
    label: 'Built-in',
    hint: 'In-process voice, HTTP PCM',
  },
  gemini: {
    label: 'Gemini Live',
    hint: 'Speech-to-speech over LiveKit',
  },
};

function storedEngine() {
  return getStored(ENGINE_STORAGE_KEY) === 'gemini' ? 'gemini' : 'builtin';
}

function newRoomName() {
  const slug = Math.random().toString(36).slice(2, 8);
  return `float-live-${Date.now().toString(36)}-${slug}`;
}

/**
 * Live audio mode: speak into the microphone, watch the avatar answer in
 * real time.
 *
 * The heavy lifting happens on the server: `POST /livekit/conversation/start`
 * spins up `FloatVideoGenerator` + `AvatarRunner`, this component posts 16 kHz
 * PCM microphone chunks, and the returned LiveKit room is watched like any
 * other stream.
 */
export default function LiveAudioStudio({
  imagePreview,
  imageBase64,
  imageFormat = 'jpg',
  onToast,
  onOpenStudio,
  portraitPicker,
}) {
  const { liveKit } = useBackendStatus();
  const [phase, setPhase] = useState('idle');
  const [error, setError] = useState(null);
  const [muted, setMuted] = useState(false);
  const [level, setLevel] = useState(0);
  const [room, setRoom] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [engine, setEngine] = useState(storedEngine);
  // Which engine the running session actually uses; the selector may change
  // while idle, but muting must follow the live session.
  const [activeEngine, setActiveEngine] = useState('builtin');
  // Autoplay with sound can be rejected even after a click; offer a retry.
  const [needsGesture, setNeedsGesture] = useState(false);
  // The portrait stays visible behind the <video> until it actually plays:
  // the avatar publishes its tracks instantly but renders no frames until
  // speech arrives, so without the backdrop the stage is a black rectangle.
  const [videoStarted, setVideoStarted] = useState(false);
  // Gemini Live only: whether the worker has actually joined the room.
  const [agentConnected, setAgentConnected] = useState(null);
  const [agentWarning, setAgentWarning] = useState(null);
  const isLive = phase === 'live';

  const videoRef = useRef(null);
  const roomRef = useRef(null);
  const viewerRef = useRef(null);
  const micRef = useRef(null);
  const micMeterRef = useRef(null);
  const mountedRef = useRef(true);
  // Aborts a slow /conversation/start when the user ends the session (or
  // leaves the page) before the server finishes warming up.
  const startAbortRef = useRef(null);
  // Every start/stop bumps this; a late await from an older start must not
  // resurrect a session the user already ended.
  const sessionRef = useRef(0);
  const stoppingRef = useRef(false);
  const startingRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      sessionRef.current += 1;
      startAbortRef.current?.abort?.();
      startAbortRef.current = null;
      const activeRoom = roomRef.current;
      roomRef.current = null;
      try {
        micMeterRef.current?.();
      } catch {
        /* ignore */
      }
      micMeterRef.current = null;
      try {
        micRef.current?.stop({ flush: false });
      } catch {
        /* ignore */
      }
      try {
        viewerRef.current?.disconnect?.();
      } catch {
        /* ignore */
      }
      if (activeRoom) stopConversation(activeRoom).catch(() => {});
    };
  }, []);

  // Elapsed timer while on air.
  useEffect(() => {
    if (phase !== 'live') return undefined;
    const started = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 1000);
    return () => clearInterval(timer);
  }, [phase]);

  // The portrait stays on the stage until the avatar's first REAL video frame
  // renders. 'playing' is not enough: the tracks are attached to one element,
  // so audio alone can satisfy the browser while the video stays black.
  // requestVideoFrameCallback only fires on actual decoded frames; fall back
  // to 'playing' when the browser does not support it (see onPlaying below).
  useEffect(() => {
    if (!isLive) {
      setVideoStarted(false);
      return undefined;
    }
    const video = videoRef.current;
    if (!video || typeof video.requestVideoFrameCallback !== 'function') {
      return undefined;
    }
    let cancelled = false;
    const onFirstFrame = (_now, _meta) => {
      if (!cancelled) setVideoStarted(true);
    };
    const register = () => video.requestVideoFrameCallback(onFirstFrame);
    register();
    return () => {
      cancelled = true;
    };
  }, [isLive, room]);

  // Poll the agent connection while a Gemini session is on air. 'Live' only
  // means the avatar's tracks exist: until the worker joins, nothing speaks
  // and the room is silent, so say so instead of a black stage.
  useEffect(() => {
    if (phase !== 'live' || activeEngine !== 'gemini') {
      setAgentConnected(null);
      setAgentWarning(null);
      return undefined;
    }
    let cancelled = false;
    let timer = null;
    const waitingSince = Date.now();
    const poll = async () => {
      const roomNow = roomRef.current;
      if (!roomNow) return;
      const status = await fetchConversationStatus(roomNow);
      if (cancelled || roomRef.current !== roomNow) return;

      if (status?.active && status?.agent_connected) {
        setAgentConnected(true);
        setAgentWarning(null);
      } else {
        setAgentConnected(false);
        // Do not cry wolf on the first poll: warming the worker up is normal.
        if (Date.now() - waitingSince > 20000) {
          setAgentWarning(
            status && status.active === false
              ? 'The conversation session ended on the server. End and restart the session.'
              : 'The Gemini agent has not joined the room yet. Check that the livekit-agents worker is running and its logs for errors.'
          );
        }
      }
      timer = setTimeout(poll, 3000);
    };
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [phase, activeEngine, room]);

  const teardownLocal = async ({ flush = true } = {}) => {
    if (micMeterRef.current) {
      try {
        micMeterRef.current();
      } catch {
        /* analyser already closed */
      }
      micMeterRef.current = null;
    }
    if (micRef.current) {
      const mic = micRef.current;
      micRef.current = null;
      await mic.stop({ flush }).catch(() => {});
    }
    if (viewerRef.current) {
      try {
        viewerRef.current.disconnect?.();
      } catch {
        /* already gone */
      }
      viewerRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      try {
        video.srcObject?.getTracks?.().forEach((track) => track.stop());
      } catch {
        /* ignore */
      }
      video.srcObject = null;
    }
  };

  // Leaving the page must not leave a GPU session running.
  useEffect(() => {
    const onUnload = () => {
      const activeRoom = roomRef.current;
      if (!activeRoom) return;
      try {
        micRef.current?.stop({ flush: false });
      } catch {
        /* ignore */
      }
      try {
        // keepalive lets the request outlive the page; sendBeacon with JSON
        // would trigger a CORS preflight it cannot survive.
        fetch(`${getApiUrl()}/livekit/conversation/stop`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ room: activeRoom }),
          keepalive: true,
        }).catch(() => {});
      } catch {
        /* ignore */
      }
    };
    window.addEventListener('beforeunload', onUnload);
    return () => window.removeEventListener('beforeunload', onUnload);
  }, []);

  const resumePlayback = async () => {
    const video = videoRef.current;
    try {
      await video?.play();
      setNeedsGesture(false);
    } catch (err) {
      console.warn('Playback is still blocked', err);
      onToast?.('Your browser is still blocking playback. Tap the video to allow sound.', 'error');
    }
  };

  const endSession = async ({ silent = false } = {}) => {
    if (stoppingRef.current) return;
    stoppingRef.current = true;
    startingRef.current = false;
    // Invalidate any in-flight start; its late awaits clean themselves up.
    sessionRef.current += 1;
    startAbortRef.current?.abort?.();
    startAbortRef.current = null;
    const activeRoom = roomRef.current;
    roomRef.current = null;
    setPhase('stopping');
    setNeedsGesture(false);
    setVideoStarted(false);
    // Local teardown can wait on a network flush; never leave the UI stuck on
    // "Ending…" because a chunk POST is timing out. The promises keep running.
    const bounded = (promise) => Promise.race([
      promise,
      new Promise((resolve) => setTimeout(resolve, 3000)),
    ]);
    await bounded(Promise.all([
      teardownLocal({ flush: true }),
      activeRoom ? stopConversation(activeRoom).catch(() => {}) : Promise.resolve(),
    ]));
    if (!mountedRef.current) {
      stoppingRef.current = false;
      return;
    }
    setRoom(null);
    setPhase('idle');
    setLevel(0);
    setMuted(false);
    setError(null);
    setElapsed(0);
    setActiveEngine('builtin');
    setAgentConnected(null);
    setAgentWarning(null);
    stoppingRef.current = false;
    if (!silent) onToast?.('Live session ended.', 'info');
  };

  const startSession = async () => {
    if (startingRef.current) return;
    if (phase === 'starting' || phase === 'stopping' || phase === 'live') return;
    if (!imageBase64) {
      onToast?.('Choose an avatar portrait first.', 'error');
      return;
    }

    startingRef.current = true;
    const token = ++sessionRef.current;
    const controller = new AbortController();
    startAbortRef.current = controller;
    const isActive = () => mountedRef.current && sessionRef.current === token;
    let startedRoom = null;
    setError(null);
    setAgentConnected(null);
    setAgentWarning(null);
    setNeedsGesture(false);
    setVideoStarted(false);
    setPhase('starting');
    const targetRoom = newRoomName();
    try {
      const started = await startConversation({
        room: targetRoom,
        imageBase64,
        imageFormat: imageFormat || 'jpg',
        agent: engine,
        geminiApiKey: engine === 'gemini' ? getGeminiApiKey() : null,
        geminiModel: engine === 'gemini' ? getGeminiModel() : null,
        signal: controller.signal,
      });
      startedRoom = started.room;
      const mode = started.agent === 'gemini' ? 'gemini' : 'builtin';
      if (isActive()) setActiveEngine(mode);

      // The user may have pressed End while the server was warming up.
      if (!isActive()) {
        await stopConversation(startedRoom).catch(() => {});
        return;
      }
      roomRef.current = started.room;
      setRoom(started.room);

      const viewer = await connectLiveKitSession({
        room: started.room,
        videoElement: videoRef.current,
        connectTimeoutMs: 20000,
        trackTimeoutMs: 30000,
        // Gemini Live listens to the room microphone; the built-in agent gets
        // PCM over HTTP instead, so it never needs a publish grant.
        publishMicrophone: mode === 'gemini',
        onStatus: (status) => {
          if (!isActive()) return;
          if (status === 'error') {
            setPhase((current) => (current === 'live' ? current : 'error'));
            setError('The LiveKit room could not be reached.');
          }
          if (status === 'done' || status === 'error') {
            // Publisher stopped: tear down the local side, but never loop.
            const endedRoom = roomRef.current;
            roomRef.current = null;
            teardownLocal({ flush: false }).catch(() => {});
            setNeedsGesture(false);
            setPhase((current) => {
              if (current === 'starting') return 'error';
              if (current === 'live') return 'idle';
              return current;
            });
            setRoom(null);
            setMuted(false);
            setElapsed(0);
            setLevel(0);
            setActiveEngine('builtin');
            setAgentConnected(null);
            setAgentWarning(null);
            if (status === 'done') {
              setError(null);
              onToast?.('The live stream ended.', 'info');
            }
            // Release the GPU session even if the publisher died on its own.
            if (endedRoom) stopConversation(endedRoom).catch(() => {});
          }
        },
      });
      viewerRef.current = viewer;

      // The room may have ended while we were connecting; do not start a mic
      // against a dead session.
      if (!isActive() || roomRef.current !== started.room) {
        viewer.disconnect?.();
        await stopConversation(started.room).catch(() => {});
        // The publisher died before we were live: the phase was flipped to
        // 'error' by onStatus, so give it something to show.
        if (isActive() && !roomRef.current) {
          setError((current) => current || 'The avatar stream ended before the session went live.');
        }
        return;
      }

      if (mode === 'gemini') {
        // The microphone is already published by connectLiveKitSession; the
        // meter only drives the UI, the agent consumes the track itself.
        micMeterRef.current = createMicrophoneMeter(viewer.room, setLevel);
      } else {
        const mic = new MicStreamer({
          room: started.room,
          onError: (message) => onToast?.(message, 'error'),
          onLevel: setLevel,
        });
        await mic.start();
        micRef.current = mic;
      }

      if (!isActive() || roomRef.current !== started.room) {
        await teardownLocal({ flush: false });
        viewer.disconnect?.();
        await stopConversation(started.room).catch(() => {});
        return;
      }
      setPhase('live');
      setElapsed(0);
      onToast?.(
        mode === 'gemini' ? 'Gemini Live is on air — start speaking.' : 'Live avatar is on air — start speaking.',
        'success'
      );
      // The audio track may unmute the element after the click that started
      // the session, so autoplay can still be rejected. Surface a retry.
      videoRef.current?.play?.().catch(() => setNeedsGesture(true));
    } catch (err) {
      console.error('Live session failed', err);
      if (startedRoom) await stopConversation(startedRoom).catch(() => {});
      if (!isActive()) return;
      const message =
        err?.name === 'NotAllowedError' || err?.name === 'SecurityError'
          ? 'Microphone access was blocked. Allow the microphone in your browser and try again.'
          : (err?.message || 'Could not start the live session.');
      setError(message);
      setPhase('error');
      await teardownLocal({ flush: false });
      roomRef.current = null;
      if (mountedRef.current) setRoom(null);
      setMuted(false);
      setElapsed(0);
      setLevel(0);
      setActiveEngine('builtin');
      setAgentConnected(null);
      setAgentWarning(null);
      onToast?.(message, 'error');
    } finally {
      if (startAbortRef.current === controller) startAbortRef.current = null;
      if (sessionRef.current === token) startingRef.current = false;
    }
  };

  const chooseEngine = (next) => {
    if (next === engine || startingRef.current || stoppingRef.current || roomRef.current) return;
    setEngine(next);
    setStored(ENGINE_STORAGE_KEY, next);
  };

  const toggleMute = () => {
    const next = !muted;
    setMuted(next);
    if (activeEngine === 'gemini') {
      const liveRoom = viewerRef.current?.room;
      if (liveRoom) {
        setRoomMicrophoneMuted(liveRoom, next).catch(() => {
          onToast?.('Could not change the microphone state.', 'error');
        });
      }
    } else {
      micRef.current?.setMuted(next);
    }
    onToast?.(next ? 'Microphone muted.' : 'Microphone live.', 'info');
  };

  const interrupt = async () => {
    if (!roomRef.current) return;
    await interruptConversation(roomRef.current).catch(() => {});
    onToast?.('Avatar interrupted.', 'info');
  };

  const configured = liveKit?.configured;
  const backendReachable = liveKit?.reachable !== false;
  const busy = phase === 'starting' || phase === 'stopping';
  const minutes = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const seconds = String(Math.floor(elapsed % 60)).padStart(2, '0');

  return (
    <main
      className="live-workspace"
      id="mode-panel-live"
      role="tabpanel"
      aria-labelledby="mode-tab-live"
    >
      <aside className="live-sidebar">
        {portraitPicker || (
          <section className="glass-panel live-pick-card">
            <div className="section-header">
              <div className="section-title-wrap">
                <span className="step-index">1</span>
                <h2 className="section-title">Avatar portrait</h2>
              </div>
              <span className="muted-label">one photo, front-facing</span>
            </div>
            <div className="live-portrait-preview">
              {imagePreview ? (
                <img src={imagePreview} alt="Selected avatar portrait" />
              ) : (
                <p className="live-portrait-empty">Choose a front-facing portrait to begin.</p>
              )}
            </div>
          </section>
        )}

        <section className="glass-panel live-pick-card">
          <div className="section-header">
            <div className="section-title-wrap">
              <span className="step-index">2</span>
              <h2 className="section-title">Microphone</h2>
            </div>
            <span className={`live-mini-pill ${isLive && !muted ? 'is-on' : ''}`}>
              <span className="live-dot" />
              {isLive ? (muted ? 'Muted' : 'On air') : 'Idle'}
            </span>
          </div>

          <div className="live-engine-row" role="group" aria-label="Conversation engine">
            {Object.entries(ENGINES).map(([key, meta]) => (
              <button
                key={key}
                type="button"
                className={`live-engine-option ${engine === key ? 'is-active' : ''}`}
                onClick={() => chooseEngine(key)}
                disabled={isLive || busy}
                aria-pressed={engine === key}
                title={meta.hint}
              >
                <span className="live-engine-label">{meta.label}</span>
                <span className="live-engine-hint">{meta.hint}</span>
              </button>
            ))}
          </div>
          {engine === 'gemini' && (liveKit?.warnings?.length || 0) > 0 && (
            <div className="live-checks">
              {liveKit.warnings.map((message, index) => (
                <p key={index} className="inline-warning">
                  <TriangleAlert size={14} />
                  {message}
                </p>
              ))}
            </div>
          )}
          {isLive && activeEngine === 'gemini' && agentWarning && (
            <div className="live-checks">
              <p className="inline-warning">
                <TriangleAlert size={14} />
                {agentWarning}
              </p>
            </div>
          )}

          <div className={`live-level ${isLive && !muted ? 'is-active' : ''}`} aria-hidden="true">
            {Array.from({ length: LEVEL_BARS }).map((_, index) => {
              // A stable pseudo-random silhouette keeps the idle state alive
              // without pretending it is detecting sound.
              const idle = 0.18 + 0.22 * Math.abs(Math.sin((index + 1) * 1.7));
              const height = isLive && !muted ? Math.max(0.08, level * (0.55 + idle)) : idle * 0.5;
              return (
                <span
                  key={index}
                  style={{ height: `${Math.min(100, height * 100)}%` }}
                />
              );
            })}
          </div>

          <p className="live-help">
            {isLive
              ? activeEngine === 'gemini'
                ? 'Speak naturally. Gemini answers speech-to-speech and the avatar lip-syncs the reply.'
                : 'Speak naturally. The avatar renders a 2-second window and keeps up in real time.'
              : engine === 'gemini'
                ? 'Gemini Live listens to your microphone through LiveKit and answers in its own voice.'
                : 'Your mic is only used for the avatar. Nothing is recorded or stored.'}
          </p>

          <div className="live-start-row">
            {!isLive ? (
              <button
                type="button"
                className="btn-primary btn-live"
                onClick={startSession}
                disabled={busy || !configured || !imageBase64}
              >
                {busy ? <span className="spinner-dot animate-spin" /> : <Mic size={18} />}
                {phase === 'starting' ? 'Starting…' : phase === 'stopping' ? 'Ending…' : 'Go Live'}
              </button>
            ) : (
              <button type="button" className="btn-danger btn-live" onClick={() => endSession()}>
                <Square size={16} fill="currentColor" />
                End session
              </button>
            )}
            {isLive && (
              <span className="live-timer" aria-live="polite">
                {minutes}:{seconds}
              </span>
            )}
          </div>

          {!configured && liveKit !== null && (
            <p className="inline-warning">
              <TriangleAlert size={14} />
              {backendReachable
                ? 'LiveKit is not configured. Add the project URL, API key and secret on the server.'
                : 'Cannot reach the backend. Check that the API is running, then reload.'}
            </p>
          )}
          {error && (
            <p className="inline-error">
              <TriangleAlert size={14} />
              {error}
            </p>
          )}
        </section>

        <section className="glass-panel live-tips">
          <h3 className="live-tips-title">
            <Zap size={15} /> For the smoothest session
          </h3>
          <ul>
            <li>Use a quiet room and a close microphone.</li>
            <li>Keep sampling steps low for faster replies.</li>
            <li>Press <strong>Interrupt</strong> to stop a reply instantly.</li>
          </ul>
        </section>
      </aside>

      <section className="glass-panel live-stage-panel">
        <div className="live-stage-header">
          <div>
            <div className="live-stage-eyebrow">
              <Radio size={13} />
              Live audio
            </div>
            <h2 className="live-stage-title">
              {isLive ? 'Your avatar is listening' : 'Real-time talking portrait'}
            </h2>
          </div>
          <span className={`live-state live-state--${phase}`} aria-live="polite">
            <span className="live-state-dot" />
            {PHASE_LABEL[phase]}
          </span>
        </div>

        <div className={`live-video ${isLive ? 'is-live' : ''}`}>
          {isLive && imagePreview && !videoStarted && (
            <img
              src={imagePreview}
              alt=""
              aria-hidden="true"
              className="live-video-backdrop"
            />
          )}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="live-video-el"
            onPlaying={() => {
              // Without requestVideoFrameCallback this is the best signal for
              // "rendering started"; with it, only a real video frame hides
              // the portrait (audio alone can fire 'playing' on a black stage).
              if (typeof videoRef.current?.requestVideoFrameCallback !== 'function') {
                setVideoStarted(true);
              }
              setNeedsGesture(false);
            }}
          />
          {isLive && needsGesture && (
            <button type="button" className="live-play-overlay" onClick={resumePlayback}>
              <Play size={18} fill="currentColor" />
              <span>Tap to start playback with sound</span>
            </button>
          )}
          {!isLive && (
            <div className="live-video-placeholder">
              {configured ? (
                <>
                  <div className="live-orb">
                    <AudioLines size={34} />
                  </div>
                  <h3>Start a live session</h3>
                  <p>
                    Pick a portrait, press <strong>Go Live</strong>, and speak. The avatar
                    answers with lip-sync in real time.
                  </p>
                  <div className="live-feature-row">
                    <span><Wand2 size={13} /> Live expressions</span>
                    <span><Zap size={13} /> Low-latency GPU</span>
                    <span><ShieldCheck size={13} /> Private by default</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="live-orb is-muted">
                    <Radio size={34} />
                  </div>
                  <h3>{backendReachable ? 'LiveKit needs to be configured' : 'Backend unavailable'}</h3>
                  <p>
                    {backendReachable
                      ? 'Live audio uses a LiveKit SFU for the avatar’s video and audio. Add the project URL, API key and secret on the server, then this panel comes alive.'
                      : 'The Studio cannot reach its API, so a live session cannot start. Check that the backend is running, then reload the page.'}
                  </p>
                  {backendReachable && (
                    <code className="live-code">fast_float/config/secerts.toml → [livekit]</code>
                  )}
                  <button type="button" className="btn-secondary" onClick={onOpenStudio}>
                    Use the studio instead
                  </button>
                </>
              )}
            </div>
          )}

          {isLive && (
            <div className="live-hud">
              <span className="live-badge">
                <span className="live-badge-dot" /> LIVE
              </span>
              <div className="live-hud-side">
                {activeEngine === 'gemini' && agentConnected !== null && (
                  <span
                    className={`live-agent-chip ${agentConnected ? 'is-connected' : 'is-waiting'}`}
                    title={agentWarning || (agentConnected
                      ? 'The Gemini Live worker is in the room.'
                      : 'Waiting for the Gemini Live worker to join the room.')}
                  >
                    {agentConnected ? 'Gemini agent connected' : 'Waiting for the Gemini agent…'}
                  </span>
                )}
                <span className="live-hud-room">{room}</span>
              </div>
            </div>
          )}
        </div>

        <div className="live-controls">
          <button
            type="button"
            className={`btn-secondary ${muted ? 'is-muted' : ''}`}
            onClick={toggleMute}
            disabled={!isLive}
            aria-pressed={muted}
            aria-label={muted ? 'Unmute microphone' : 'Mute microphone'}
          >
            {muted ? <MicOff size={16} aria-hidden="true" /> : <Mic size={16} aria-hidden="true" />}
            {muted ? 'Unmute' : 'Mute'}
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={interrupt}
            disabled={!isLive}
          >
            <CircleStop size={16} aria-hidden="true" />
            Interrupt
          </button>
          <button
            type="button"
            className="btn-secondary btn-live-end"
            onClick={() => endSession()}
            disabled={!isLive && phase !== 'starting'}
          >
            <Sparkles size={16} aria-hidden="true" />
            End &amp; clear
          </button>
          <span className="live-controls-hint">
            {isLive && activeEngine === 'gemini'
              ? 'Gemini Live owns the voice — the avatar only renders it.'
              : 'Audio goes straight to the GPU renderer — no clips are stored.'}
          </span>
        </div>
      </section>
    </main>
  );
}
