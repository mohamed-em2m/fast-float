/**
 * Portrait presets, discovered from the filesystem instead of a hard-coded list.
 *
 * Every image dropped into `src/assets/samples/` becomes a button in the
 * Studio's "Quick Samples" row. The previous list was written by hand and
 * drifted: it still pointed at `portrait2.jpg` after that file was deleted, so
 * the row rendered a broken thumbnail and a button that could only fail.
 *
 * `import.meta.glob` is resolved by Vite at build time (and by Vitest in tests),
 * so the URLs are the bundled, hashed asset URLs and the list is available
 * synchronously - the first sample can therefore also serve as the default
 * portrait, before any network round trip.
 */
const SAMPLE_MODULES = import.meta.glob(
  '../assets/samples/*.{jpg,jpeg,png,webp,avif}',
  { eager: true, query: '?url', import: 'default' },
);

const EXTENSION = /\.[^.]+$/;

/** `portrait-3` / `portrait3` / `some_face.jpg` -> "Portrait 3" / "Some Face". */
export function labelFromId(id) {
  const spaced = id
    .replace(/[-_]+/g, ' ')
    .replace(/([a-z])(\d)/gi, '$1 $2')
    .trim();
  return spaced.replace(/\b\w/g, (char) => char.toUpperCase()) || 'Sample';
}

/**
 * Sorted by file name (numbers compared numerically, so `portrait10` follows
 * `portrait9` rather than `portrait1`).
 */
export const SAMPLE_PORTRAITS = Object.entries(SAMPLE_MODULES)
  .map(([path, url]) => {
    const file = path.split('/').pop() || '';
    const id = file.replace(EXTENSION, '');
    return { id, name: labelFromId(id), url };
  })
  .sort((a, b) => a.id.localeCompare(b.id, 'en', { numeric: true }));

/** First discovered portrait, or `null` when the samples folder is empty. */
export const DEFAULT_PORTRAIT_URL = SAMPLE_PORTRAITS[0]?.url ?? null;
