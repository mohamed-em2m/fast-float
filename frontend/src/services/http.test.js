import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchJson, fetchWithTimeout, HttpError, TimeoutError } from './http';

function jsonResponse(body, { status = 200, statusText = 'OK' } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('fetchWithTimeout', () => {
  it('aborts a hung request and reports a TimeoutError', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn((_url, init) => new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => {
        reject(new DOMException('Aborted', 'AbortError'));
      });
    }));
    vi.stubGlobal('fetch', fetchMock);

    const request = fetchWithTimeout('http://api.test/health', {}, 1000);
    const expectation = expect(request).rejects.toBeInstanceOf(TimeoutError);
    await vi.advanceTimersByTimeAsync(1000);
    await expectation;
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('forwards an external abort signal', async () => {
    const controller = new AbortController();
    const fetchMock = vi.fn((_url, init) => new Promise((_resolve, reject) => {
      const fail = () => reject(new DOMException('Aborted', 'AbortError'));
      if (init.signal.aborted) {
        fail();
        return;
      }
      init.signal.addEventListener('abort', fail);
    }));
    vi.stubGlobal('fetch', fetchMock);

    const request = fetchWithTimeout('http://api.test/slow', { signal: controller.signal }, 5000);
    controller.abort();
    await expect(request).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe('fetchJson', () => {
  it('parses a JSON body', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ status: 'healthy' })));
    await expect(fetchJson('http://api.test/health')).resolves.toEqual({ status: 'healthy' });
  });

  it('normalizes HTTP errors and keeps the backend detail', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(
      { detail: 'Server busy — try again shortly' },
      { status: 429, statusText: 'Too Many Requests' }
    )));

    await expect(fetchJson('http://api.test/generate')).rejects.toMatchObject({
      name: 'HttpError',
      status: 429,
      message: 'Server busy — try again shortly',
    });
  });

  it('rejects malformed JSON with an HttpError', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({}),
      text: async () => '<html>not json</html>',
    })));

    await expect(fetchJson('http://api.test/x')).rejects.toBeInstanceOf(HttpError);
  });

  it('returns null for empty responses', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      status: 204,
      statusText: 'No Content',
      json: async () => null,
      text: async () => '',
    })));

    await expect(fetchJson('http://api.test/x')).resolves.toBeNull();
  });
});
