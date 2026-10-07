/* HTTP layer for the reid-gallery API. Every endpoint is wrapped so the
 * views never touch fetch() directly and error handling stays uniform. */

async function request(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      if (j && j.detail) detail = j.detail;
    } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

const get = (path) => request(path);

const post = (path, body) =>
  request(path, { method: "POST", body: body ?? {} });

const patch = (path, body) =>
  request(path, { method: "PATCH", body });

const del = (path) => request(path, { method: "DELETE" });

export const api = {
  get,
  post,
  patch,
  del,

  status: () => get("/api/status"),
  config: () => get("/api/config"),

  photos: (offset = 0, limit = 60) =>
    get(`/api/photos?offset=${offset}&limit=${limit}`),
  photo: (id) => get(`/api/photos/${id}`),
  deletePhoto: (id) => del(`/api/photos/${id}`),

  persons: () => get("/api/persons"),
  person: (id, offset = 0, limit = 120) =>
    get(`/api/persons/${id}?offset=${offset}&limit=${limit}`),
  personFaces: (id, limit = 60) =>
    get(`/api/persons/${id}/faces?limit=${limit}`),
  renamePerson: (id, body) => patch(`/api/persons/${id}`, body),
  mergePerson: (id, target_id) =>
    post(`/api/persons/${id}/merge`, { target_id }),
  resetPerson: (id) => post(`/api/persons/${id}/reset`, {}),
  breakup: (id, apply) => post(`/api/persons/${id}/breakup`, { apply }),

  suggest: (kind, refId) => get(`/api/suggest/${kind}/${refId}`),

  review: (limit = 24, targetId) =>
    get(`/api/review?limit=${limit}` + (targetId ? `&target_id=${targetId}` : "")),
  reviewMerge: (body) => post("/api/review/merge", body),
  reviewSkip: (body) => post("/api/review/skip", body),
  reviewReject: (body) => post("/api/review/reject", body),
  reviewUndo: () => post("/api/review/undo", {}),
  reviewDecided: (targetId) =>
    get(`/api/review/decided` + (targetId ? `?target_id=${targetId}` : "")),
  reviewReopen: (a, b) => post("/api/review/reopen", { a, b }),

  label: (kind, refId, body) => post(`/api/label/${kind}/${refId}`, body),

  adminReset: (mode) => post("/api/admin/reset", { mode }),
  scan: (path) => post("/api/scan", { path }),
  recluster: () => post("/api/recluster", {}),

  labelsExport: "/api/labels/export",

  upload: async (files) => {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const j = await res.json();
        if (j && j.detail) detail = j.detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  },
};

/* Person identity helpers. The display label and the DB id are kept
 * strictly separate: labels come from the server (name or "person #seq"),
 * routes and API calls always use the DB id. */

export function personLabel(p) {
  if (!p) return "";
  if (p.name && p.name.trim()) return p.name;
  if (p.display) return p.display;
  return p.seq != null ? `person #${p.seq}` : "";
}

export function personHref(p) {
  return `/person/${p.id ?? p.person_id}`;
}
