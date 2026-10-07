/* Reactive store. Vue components read this and re-render automatically when
 * it changes - this is what eliminates the manual sync bugs the old UI had
 * (avatar/photos drifting apart, stale person lists after re-cluster). */

const { reactive } = Vue;
import { api } from "./api.js";

export const store = reactive({
  status: {
    photos: 0,
    processed: 0,
    pending: 0,
    persons: 0,
    faces: 0,
    detections: 0,
    worker: { queue: 0, processed: 0, failed: 0, last_error: null },
  },
  persons: [],          // [{id, name, display, seq, pinned, photos, cover}]
  personsLoading: false,
  peopleSort: "count-desc",
  peopleSearch: "",
  busy: false,          // global action-in-progress flag
  lastError: null,
  // Ente mobile: long-press enters selection, the tab bar steps aside for
  // the selection action bar. Shared here because the shell owns the bar.
  selection: { kind: null, ids: [] },   // kind: 'person' | 'photo'
});

export const selection = {
  get active() {
    return store.selection.ids.length > 0;
  },
  // the templates work with selection.ids / selection.kind directly, so the
  // helper has to expose the underlying state, not just the operations
  get ids() {
    return store.selection.ids;
  },
  get kind() {
    return store.selection.kind;
  },
  has(id) {
    return store.selection.ids.includes(id);
  },
  enter(kind, id) {
    store.selection.kind = kind;
    store.selection.ids = [id];
  },
  toggle(id) {
    const i = store.selection.ids.indexOf(id);
    if (i >= 0) store.selection.ids.splice(i, 1);
    else store.selection.ids.push(id);
  },
  clear() {
    store.selection.kind = null;
    store.selection.ids = [];
  },
};

export async function refreshStatus() {
  try {
    store.status = await api.status();
  } catch (e) {
    store.lastError = e.message;
  }
}

export async function refreshPersons() {
  store.personsLoading = true;
  try {
    const r = await api.persons();
    store.persons = r.persons || [];
  } catch (e) {
    store.lastError = e.message;
  } finally {
    store.personsLoading = false;
  }
}

/* Sorting + filtering for the people grid. Ordering follows the Ente
 * convention: pinned first, then named before unnamed, then size. */
export function visiblePersons() {
  const q = store.peopleSearch.trim().toLowerCase();
  let list = store.persons.slice();
  if (q) {
    list = list.filter((p) =>
      (p.name || "").toLowerCase().includes(q) ||
      (p.display || "").toLowerCase().includes(q));
  }
  const sort = store.peopleSort;
  if (sort === "name-asc" || sort === "name-desc") {
    const dir = sort === "name-asc" ? 1 : -1;
    list.sort((a, b) => {
      const an = (a.name || "").toLowerCase();
      const bn = (b.name || "").toLowerCase();
      if (an && !bn) return -1;
      if (!an && bn) return 1;
      if (an && bn) return an < bn ? -dir : an > bn ? dir : 0;
      return (a.seq || 0) - (b.seq || 0);
    });
  } else {
    const dir = sort === "count-asc" ? 1 : -1;
    list.sort((a, b) => dir * ((a.photos || 0) - (b.photos || 0)));
  }
  list.sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0));
  return list;
}
