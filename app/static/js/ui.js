/* Small shared UI wheels and helpers. */

export function toast(message, type = "info", ms = 2600) {
  if (!window.Toastify) return;
  const map = {
    success: "#3fb950",
    error: "#f85149",
    info: "#5b8cff",
    warn: "#d29922",
  };
  Toastify({
    text: message,
    duration: ms,
    gravity: "top",
    position: "right",
    style: { background: map[type] || map.info, color: "#fff" },
  }).showToast();
}

/* Webview-safe two-step confirm. window.confirm is often swallowed by
 * embedded browsers (Telegram etc.), so the button arms first and the
 * second click confirms. Resolves true only on the second click. */
export function armConfirm(button, label = "Click again to confirm") {
  return new Promise((resolve) => {
    if (button.dataset.armed === "1") {
      button.dataset.armed = "";
      button.textContent = button.dataset.label || button.textContent;
      resolve(true);
      return;
    }
    button.dataset.label = button.dataset.label || button.textContent;
    button.dataset.armed = "1";
    const original = button.dataset.label;
    button.textContent = label;
    const timer = setTimeout(() => {
      if (button.dataset.armed === "1") {
        button.dataset.armed = "";
        button.textContent = original;
        resolve(false);
      }
    }, 4000);
    button.addEventListener(
      "click",
      function onSecond() {
        if (button.dataset.armed === "1") {
          clearTimeout(timer);
          button.dataset.armed = "";
          button.textContent = original;
          button.removeEventListener("click", onSecond, true);
          resolve(true);
        }
      },
      true
    );
  });
}

export function fmtDate(ts) {
  if (!ts) return "Unknown date";
  const d = new Date(ts * 1000);
  if (isNaN(d)) return "Unknown date";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function fmtDay(ts) {
  return fmtDate(ts);
}

export function pct(x) {
  if (x == null) return "-";
  return Math.round(x * 100) + "%";
}

/* Photos into day buckets, newest first (server already sorts desc). */
export function groupByDay(photos) {
  const out = [];
  let cur = null;
  for (const p of photos) {
    const day = fmtDay(p.taken_at);
    if (!cur || cur.day !== day) {
      cur = { day, items: [] };
      out.push(cur);
    }
    cur.items.push(p);
  }
  return out;
}
