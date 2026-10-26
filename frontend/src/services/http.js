/**
 * Small fetch helpers shared by every backend client.
 *
 * Every call is bounded: a stopped/hung backend must not leave the Studio on
 * "Connecting…" forever, and callers get errors that are easy to distinguish
 * (timeout vs. HTTP failure vs. an abort triggered by unmount).
 */

/** The server answered, but not with 2xx. */
export class HttpError extends Error {
  constructor(message, { status = 0, url = '', body = null } = {}) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

/** The request exceeded its deadline and was aborted locally. */
export class TimeoutError extends Error {
  constructor(message = 'Request timed out', { url = '', timeoutMs = 0 } = {}) {
    super(message);
    this.name = 'TimeoutError';
    this.url = url;
    this.timeoutMs = timeoutMs;
  }
}

/** True for timeouts and user/unmount aborts (expected, not a crash). */
export function isAbortError(err) {
  return err?.name === 'AbortError' || err?.name === 'TimeoutError';
}

/**
 * `fetch` with a deadline and optional external abort signal.
 *
 * `init.signal` is honoured as well as `options.signal`; the external signal is
 * forwarded to the internal controller so one abort cleans everything up.
 * A timeout is rethrown as {@link TimeoutError} instead of the browser's
 * generic AbortError, so callers can show "no response in 8s".
 */
export async function fetchWithTimeout(url, init = {}, timeoutMs = 8000) {
  const { signal: externalSignal, ...rest } = init;
  const controller = new AbortController();
  let timedOut = false;

  const abortFromOutside = () => controller.abort(externalSignal.reason);
  if (externalSignal) {
    if (externalSignal.aborted) controller.abort(externalSignal.reason);
    else externalSignal.addEventListener('abort', abortFromOutside, { once: true });
  }

  const timer = timeoutMs > 0
    ? setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs)
    : null;

  try {
    return await fetch(url, { ...rest, signal: controller.signal });
  } catch (err) {
    if (timedOut) {
      throw new TimeoutError(
        `No response after ${Math.round(timeoutMs / 1000)}s`,
        { url: String(url), timeoutMs }
      );
    }
    throw err;
  } finally {
    if (timer) clearTimeout(timer);
    externalSignal?.removeEventListener?.('abort', abortFromOutside);
  }
}

/**
 * Fetch + parse JSON, normalizing HTTP errors.
 *
 * Returns `null` for 204/empty bodies. Error messages prefer the backend's
 * FastAPI `detail` field so the UI can surface a specific reason.
 */
export async function fetchJson(url, init = {}, { timeoutMs = 8000, signal } = {}) {
  const res = await fetchWithTimeout(url, { ...init, signal }, timeoutMs);
  if (!res.ok) {
    let body = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error page */
    }
    const detail = body?.detail || body?.message || body?.error;
    throw new HttpError(detail || `HTTP ${res.status} ${res.statusText}`.trim(), {
      status: res.status,
      url: String(url),
      body,
    });
  }
  if (res.status === 204) return null;
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    throw new HttpError('The server returned malformed JSON.', {
      status: res.status,
      url: String(url),
    });
  }
}
