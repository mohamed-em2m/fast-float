/**
 * Safe browser-storage helpers.
 *
 * `localStorage` is not always available: Safari private mode throws on write,
 * sandboxed iframes and hardened privacy settings can throw on *read* (even at
 * module import time), and the storage quota can be exhausted. A single
 * unguarded `localStorage.getItem()` used to blank the whole Studio before
 * React could mount, so every access goes through this module.
 *
 * Values written this session are also mirrored in memory, which keeps the UI
 * consistent when persistence is unavailable and makes a failed write behave
 * like a normal in-memory setting.
 */

/** Session-only mirror so a blocked/full localStorage never loses a setting. */
const memory = new Map();

function storage() {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage || null;
  } catch {
    return null;
  }
}

/**
 * Read a string value, or `fallback` when the key is missing/unavailable.
 * Never throws.
 */
export function getStored(key, fallback = null) {
  if (!key) return fallback;
  if (memory.has(key)) return memory.get(key);
  try {
    const value = storage()?.getItem(key);
    if (typeof value === 'string') {
      memory.set(key, value);
      return value;
    }
  } catch {
    /* unavailable or blocked */
  }
  return fallback;
}

/**
 * Persist a string value. Returns `true` when it reached localStorage, `false`
 * when only the in-memory mirror could be updated. Never throws.
 */
export function setStored(key, value) {
  if (!key) return false;
  const text = String(value);
  memory.set(key, text);
  try {
    const store = storage();
    if (!store) return false;
    store.setItem(key, text);
    return true;
  } catch {
    return false;
  }
}

/** Remove a value from both storage and the in-memory mirror. Never throws. */
export function removeStored(key) {
  if (!key) return;
  memory.delete(key);
  try {
    storage()?.removeItem(key);
  } catch {
    /* unavailable or blocked */
  }
}

/** Testing hook: forget mirrored values between cases. */
export function clearStoredMemory() {
  memory.clear();
}
