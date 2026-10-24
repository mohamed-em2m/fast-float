import React, { useEffect, useRef, useState } from 'react';
import {
  AudioLines,
  CircleStop,
  Mic,
  MicOff,
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
import {
  connectLiveKitSession,
  fetchLiveKitStatus,
  interruptConversation,
  startConversation,
  stopConversation,
} from '../services/livekit';

const PHASE_LABEL = {
  idle: 'Ready when you are',
  starting: 'Warming up the avatar…',
  live: 'Live',
  stopping: 'Ending session…',
  error: 'Session failed',
};

const LEVEL_BARS = 28;

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
  onToast,
  onOpenStudio,
  portraitPicker,
}) {
  const [liveKit, setLiveKit] = useState(null);
  const [phase, setPhase] = useState('idle');
  const [error, setError] = useState(null);
  const [muted, setMuted] = useState(false);
  const [level, setLevel] = useState(0);
  const [room, setRoom] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  const videoRef = useRef(null);
  const roomRef = useRef(null);
  const viewerRef = useRef(null);
  const micRef = useRef(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      const activeRoom = roomRef.current;
      roomRef.current = null;
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

  useEffect(() => {
    let cancelled = false;
    const check = () => {
      fetchLiveKitStatus()
        .then((status) => {
          if (!cancelled) setLiveKit(status);
        })
        .catch(() => {
          if (!cancelled) setLiveKit({ configured: false });
        });
    };
    check();
    window.addEventListener('float:api_url_changed', check);
    return () => {
      cancelled = true;
      window.removeEventListener('float:api_url_changed', check);
    };
  }, []);

  // Elapsed timer while on air.
  useEffect(() => {
    if (phase !== 'live') return undefined;
    const started = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 1000);
    return () => clearInterval(timer);
  }, [phase]);

  const teardownLocal = async ({ flush = true } = {}) => {
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

  const endSession = async ({ silent = false } = {}) => {
    const activeRoom = roomRef.current;
    setPhase('stopping');
    await teardownLocal({ flush: true });
    if (activeRoom) await stopConversation(activeRoom).catch(() => {});
    roomRef.current = null;
    if (!mountedRef.current) return;
    setRoom(null);
    setPhase('idle');
    setLevel(0);
    setMuted(false);
    setError(null);
    if (!silent) onToast?.('Live session ended.', 'info');
  };

  const startSession = async () => {
    if (!imageBase64) {
      onToast?.('Choose an avatar portrait first.', 'error');
      return;
    }
    setError(null);
    setPhase('starting');
    const targetRoom = newRoomName();
    try {
      const started = await startConversation({
        room: targetRoom,
        imageBase64,
        imageFormat: 'jpg',
      });
      roomRef.current = started.room;
      setRoom(started.room);

      const viewer = await connectLiveKitSession({
        room: started.room,
        videoElement: videoRef.current,
        connectTimeoutMs: 20000,
        trackTimeoutMs: 30000,
        onStatus: (status) => {
          if (!mountedRef.current) return;
          if (status === 'error') {
            setPhase((current) => (current === 'live' ? current : 'error'));
            setError('The LiveKit room could not be reached.');
          }
          if (status === 'done' || status === 'error') {
            // Publisher stopped: tear down the local side, but never loop.
            const endedRoom = roomRef.current;
            teardownLocal({ flush: false });
            setPhase((current) => {
              if (current === 'starting') return 'error';
              if (current === 'live') return 'idle';
              return current;
            });
            setRoom(null);
            roomRef.current = null;
            // Release the GPU session even if the publisher died on its own.
            if (endedRoom) stopConversation(endedRoom).catch(() => {});
          }
        },
      });
      viewerRef.current = viewer;

      // The room may have ended while we were connecting; do not start a mic
      // against a dead session.
      if (!mountedRef.current || roomRef.current !== started.room) {
        viewer.disconnect?.();
        await stopConversation(started.room).catch(() => {});
        return;
      }

      const mic = new MicStreamer({
        room: started.room,
        onError: (message) => onToast?.(message, 'error'),
        onLevel: setLevel,
      });
      await mic.start();
      micRef.current = mic;

      if (!mountedRef.current || roomRef.current !== started.room) {
        await mic.stop({ flush: false });
        viewer.disconnect?.();
        await stopConversation(started.room).catch(() => {});
        return;
      }
      setPhase('live');
      setElapsed(0);
      onToast?.('Live avatar is on air — start speaking.', 'success');
    } catch (err) {
      console.error('Live session failed', err);
      setError(err.message || 'Could not start the live session.');
      setPhase('error');
      await teardownLocal({ flush: false });
      const activeRoom = roomRef.current;
      roomRef.current = null;
      if (activeRoom) await stopConversation(activeRoom).catch(() => {});
      setRoom(null);
      onToast?.(err.message || 'Could not start the live session.', 'error');
    }
  };

  const toggleMute = () => {
    const next = !muted;
    setMuted(next);
    micRef.current?.setMuted(next);
    onToast?.(next ? 'Microphone muted.' : 'Microphone live.', 'info');
  };

  const interrupt = async () => {
    if (!roomRef.current) return;
    await interruptConversation(roomRef.current).catch(() => {});
    onToast?.('Avatar interrupted.', 'info');
  };

  const configured = liveKit?.configured;
  const busy = phase === 'starting' || phase === 'stopping';
  const isLive = phase === 'live';
  const minutes = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const seconds = String(Math.floor(elapsed % 60)).padStart(2, '0');

  return (
    <main className="live-workspace">
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
              <img src={imagePreview} alt="Selected avatar portrait" />
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
              ? 'Speak naturally. The avatar renders a 2-second window and keeps up in real time.'
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
              LiveKit is not configured. See the setup card on the right.
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
          <video ref={videoRef} autoPlay playsInline className="live-video-el" />
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
                  <h3>LiveKit needs to be running</h3>
                  <p>
                    Live audio uses a LiveKit SFU for the avatar&apos;s video and audio.
                    Start it once, then this panel comes alive.
                  </p>
                  <code className="live-code">docker compose -f docker-compose.livekit.yml up -d</code>
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
              <span className="live-hud-room">{room}</span>
            </div>
          )}
        </div>

        <div className="live-controls">
          <button
            type="button"
            className={`btn-secondary ${muted ? 'is-muted' : ''}`}
            onClick={toggleMute}
            disabled={!isLive}
          >
            {muted ? <MicOff size={16} /> : <Mic size={16} />}
            {muted ? 'Unmute' : 'Mute'}
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={interrupt}
            disabled={!isLive}
          >
            <CircleStop size={16} />
            Interrupt
          </button>
          <button
            type="button"
            className="btn-secondary btn-live-end"
            onClick={() => endSession()}
            disabled={!isLive && phase !== 'starting'}
          >
            <Sparkles size={16} />
            End &amp; clear
          </button>
          <span className="live-controls-hint">
            Audio goes straight to the GPU renderer — no clips are stored.
          </span>
        </div>
      </section>
    </main>
  );
}
