/**
 * Copy text to the clipboard with a fallback for insecure origins.
 *
 * `navigator.clipboard` is only available in secure contexts (HTTPS or
 * localhost). The Studio is routinely opened over plain HTTP from a LAN IP or
 * tunnel, where writing to the clipboard throws and a rejected promise would
 * otherwise show a misleading "copied" toast.
 */
export async function copyText(text) {
  const value = String(text ?? '');
  if (!value) return false;

  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText && window.isSecureContext) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    /* fall through to execCommand */
  }

  try {
    const area = document.createElement('textarea');
    area.value = value;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.top = '-1000px';
    area.style.left = '-1000px';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    area.setSelectionRange(0, value.length);
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  } catch {
    return false;
  }
}
