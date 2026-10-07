/* People grid, modelled on Ente mobile's PeopleSectionAllPage:
 *   squircle tiles (avatar + name, pinned badge), tap to open,
 *   Select button for multi-select, bottom action bar while selecting,
 *   sort + search behind the top bar.
 *
 * Selection is entered from a button, never from a long press: a press-and-
 * hold competes with scrolling and keeps firing while a page is being read.
 */

const { ref, computed, onMounted, onBeforeUnmount } = Vue;
import { api, personLabel, personHref } from "./api.js";
import { store, selection, refreshPersons, refreshStatus, visiblePersons } from "./store.js";
import { toast } from "./ui.js";

export const PeopleView = {
  name: "PeopleView",
  setup() {
    const loaded = ref(false);
    const sortOpen = ref(false);
    const sheet = ref(null);          // null | {kind, ...}
    const selectMode = ref(false);    // entered from the Select button

    onMounted(async () => {
      if (!store.persons.length) await refreshPersons();
      refreshStatus();
      loaded.value = true;
    });
    onBeforeUnmount(() => selection.clear());

    const list = computed(() => visiblePersons());
    const selected = computed(() =>
      store.persons.filter((p) => selection.has(p.id)));

    /* Tap opens the person. With the Select button pressed, tap instead
     * toggles membership - the same split Ente uses, minus the long press. */
    function onTap(p) {
      if (selectMode.value) {
        selection.toggle(p.id);
        return;
      }
      location.hash = personHref(p);
    }
    function enterSelect() {
      selectMode.value = true;
      selection.clear();
    }
    function exitSelect() {
      selectMode.value = false;
      selection.clear();
    }
    function cancelSelection() {
      exitSelect();
    }

    /* ---- selection actions (Ente people_selection_action_widget) ---- */
    function actEdit() {
      const p = selected.value[0];
      if (!p) return;
      sheet.value = { kind: "rename", id: p.id, name: p.name || "" };
    }
    function actReview() {
      const p = selected.value[0];
      if (!p) return;
      location.hash = `/review?t=${p.id}`;
    }
    function actPin() {
      const val = !selected.value.every((p) => p.pinned);
      Promise.all(selected.value.map((p) =>
        api.renamePerson(p.id, { pinned: val })))
        .then(() => { toast(val ? "Pinned" : "Unpinned", "success"); return refreshPersons(); })
        .catch((e) => toast(e.message, "error"));
    }
    function actMerge() {
      // Ente: exactly one named person -> it is the target, otherwise pick
      const named = selected.value.filter((p) => (p.name || "").trim());
      sheet.value = {
        kind: "merge",
        target: named.length === 1 ? named[0].id : (selected.value[0] || {}).id,
      };
    }
    function actReset() {
      sheet.value = { kind: "reset" };
    }

    async function confirmRename() {
      try {
        await api.renamePerson(sheet.value.id, { name: sheet.value.name.trim() });
        toast("Saved", "success");
        sheet.value = null;
        selection.clear();
        await refreshPersons();
      } catch (e) { toast(e.message, "error"); }
    }
    async function confirmMerge() {
      const target = sheet.value.target;
      const others = selection.ids.filter((id) => id !== target);
      try {
        for (const id of others) await api.mergePerson(id, target);
        toast(`Merged ${others.length} into ${personLabel(
          store.persons.find((p) => p.id === target))}`, "success");
        sheet.value = null;
        selection.clear();
        await refreshPersons();
      } catch (e) { toast(e.message, "error"); }
    }
    async function confirmReset() {
      try {
        for (const id of selection.ids) await api.resetPerson(id);
        toast("Person grouping reset", "success");
        sheet.value = null;
        selection.clear();
        await refreshPersons();
      } catch (e) { toast(e.message, "error"); }
    }

    const SORTS = [
      ["count-desc", "Most photos"], ["count-asc", "Fewest photos"],
      ["name-asc", "Name A–Z"], ["name-desc", "Name Z–A"],
    ];

    return {
      store, selection, list, selected, loaded, sortOpen, sheet, SORTS,
      personLabel, personHref, onTap, cancelSelection,
      selectMode, enterSelect, exitSelect,
      actEdit, actReview, actPin, actMerge, actReset,
      confirmRename, confirmMerge, confirmReset,
    };
  },
  template: `
  <div class="view">
    <div class="view-head">
      <h1>People <span class="count">{{ list.length }}</span></h1>
      <div class="view-head-side">
        <span class="sel-hint" v-if="selectMode">{{ selected.length }} selected</span>
        <button class="btn ghost" type="button" v-if="!selectMode"
                @click="enterSelect">Select</button>
        <button class="btn primary" type="button" v-else
                @click="exitSelect">Done</button>
        <button class="icon-btn" type="button" @click="sortOpen = !sortOpen"
                aria-label="Sort">Sort</button>
      </div>
    </div>

    <div class="sort-sheet" v-if="sortOpen">
      <button class="btn ghost" type="button" v-for="[v, label] in SORTS" :key="v"
              :class="{ primary: store.peopleSort === v }"
              @click="store.peopleSort = v; sortOpen = false">{{ label }}</button>
    </div>

    <div v-if="store.personsLoading && !loaded" class="empty">Loading people…</div>
    <div v-else-if="!list.length" class="empty">
      <h2>No people yet</h2>
      <p>Photos need to be scanned and clustered first.</p>
      <router-link class="btn primary" to="/settings">Open Settings</router-link>
    </div>

    <div class="people-grid">
      <button class="person-tile" v-for="p in list" :key="p.id" type="button"
              :class="{ selected: selection.has(p.id), selectable: selectMode }"
              @click="onTap(p)">
        <span class="avatar-wrap">
          <img class="avatar" v-if="p.cover" :src="p.cover" alt="">
          <span class="avatar" v-else></span>
          <span class="pin" v-if="p.pinned">Pinned</span>
          <span class="tick">✓</span>
        </span>
        <span class="tile-name" :class="{ unnamed: !p.name }">{{ personLabel(p) }}</span>
      </button>
    </div>

    <!-- Ente: the tab bar steps aside for this -->
    <div class="sel-bar" v-if="selection.active">
      <div class="sel-actions">
        <button class="btn" type="button" v-if="selection.ids.length === 1" @click="actEdit">Edit</button>
        <button class="btn" type="button" v-if="selection.ids.length === 1" @click="actReview">Review</button>
        <button class="btn" type="button" @click="actPin">Pin / Unpin</button>
        <button class="btn" type="button" v-if="selection.ids.length > 1" @click="actMerge">Merge</button>
        <button class="btn" type="button" @click="actReset">Reset</button>
      </div>
      <div class="sel-foot">
        <span>{{ selection.ids.length }} selected</span>
        <button class="btn ghost" type="button" @click="cancelSelection">Cancel</button>
      </div>
    </div>

    <!-- bottom sheets: the mobile confirmation pattern -->
    <div class="sheet-backdrop" v-if="sheet" @click.self="sheet = null">
      <div class="sheet">
        <template v-if="sheet.kind === 'rename'">
          <h3>Edit person</h3>
          <input class="input" type="text" v-model="sheet.name" placeholder="Enter name"
                 @keyup.enter="confirmRename">
          <div class="sheet-actions">
            <button class="btn ghost" type="button" @click="sheet = null">Cancel</button>
            <button class="btn primary" type="button" @click="confirmRename">Save</button>
          </div>
        </template>

        <template v-else-if="sheet.kind === 'merge'">
          <h3>Merge {{ selection.ids.length }} people into</h3>
          <div class="merge-targets">
            <button class="person-tile" type="button" v-for="p in selected" :key="p.id"
                    :class="{ selected: sheet.target === p.id }"
                    @click="sheet.target = p.id">
              <span class="avatar-wrap">
                <img class="avatar" v-if="p.cover" :src="p.cover" alt="">
                <span class="avatar" v-else></span>
                <span class="tick">✓</span>
              </span>
              <span class="tile-name">{{ personLabel(p) }}</span>
            </button>
          </div>
          <p class="muted">Everyone else joins <b>{{
            personLabel(selected.find(p => p.id === sheet.target)) }}</b>.</p>
          <div class="sheet-actions">
            <button class="btn ghost" type="button" @click="sheet = null">Cancel</button>
            <button class="btn primary" type="button" @click="confirmMerge">Merge</button>
          </div>
        </template>

        <template v-else-if="sheet.kind === 'reset'">
          <h3>Reset {{ selection.ids.length }} person(s)?</h3>
          <p class="muted">All person grouping will reset. Names are kept.</p>
          <div class="sheet-actions">
            <button class="btn ghost" type="button" @click="sheet = null">Cancel</button>
            <button class="btn danger" type="button" @click="confirmReset">Reset</button>
          </div>
        </template>
      </div>
    </div>
  </div>
  `,
};
