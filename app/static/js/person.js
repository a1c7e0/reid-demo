/* Person page: header actions (rename / pin / review / split / reset) and
 * the person's own photo grid. Avatar comes from the same server rule as the
 * people list, so they can never disagree. */

const { ref, onMounted, computed, watch } = Vue;
import { api, personLabel, personHref } from "./api.js";
import { store, refreshPersons, refreshStatus } from "./store.js";
import { groupByDay, toast, armConfirm } from "./ui.js";
import { PhotoViewer } from "./viewer.js";
import { ReviewCard } from "./review.js";

export const PersonView = {
  name: "PersonView",
  components: { PhotoViewer, ReviewCard },
  props: { id: { type: Number, required: true } },
  setup(props) {
    const person = ref(null);
    const loading = ref(true);
    const err = ref(null);
    const photos = ref([]);
    const faces = ref([]);
    const renaming = ref(false);
    const newName = ref("");
    const showSplit = ref(false);
    const splitGroups = ref([]);
    const viewerStart = ref(-1);
    const reviewTarget = ref(null); // person id when the per-person review is open

    async function load() {
      loading.value = true;
      err.value = null;
      try {
        const r = await api.person(props.id);
        person.value = r;
        photos.value = r.photos || [];
        const fr = await api.personFaces(props.id, 120);
        faces.value = fr.faces || [];
      } catch (e) {
        err.value = e.message;
      } finally {
        loading.value = false;
      }
    }

    function startRename() {
      newName.value = (person.value.name || "");
      renaming.value = true;
    }
    async function saveRename() {
      try {
        const r = await api.renamePerson(props.id, { name: newName.value });
        renaming.value = false;
        if (r.merged_from) {
          toast("Name already exists — merged", "warn");
          await load();
        }
        await refreshPersons();
        await load();
        toast("Renamed", "success");
      } catch (e) { toast(e.message, "error"); }
    }
    async function togglePin() {
      try {
        await api.renamePerson(props.id, { pinned: !person.value.pinned });
        person.value.pinned = !person.value.pinned;
        await refreshPersons();
        toast(person.value.pinned ? "Pinned" : "Unpinned", "success");
      } catch (e) { toast(e.message, "error"); }
    }
    async function doReset(btn) {
      const ok = await armConfirm(btn);
      if (!ok) return;
      try {
        await api.resetPerson(props.id);
        toast("Person reset", "success");
        await refreshPersons();
        router.push("/people");
      } catch (e) { toast(e.message, "error"); }
    }
    async function previewSplit() {
      try {
        const r = await api.breakup(props.id, false);
        if (r.count < 2) {
          toast("Homogeneous at this threshold — nothing to split", "info");
          return;
        }
        splitGroups.value = r.groups;
        showSplit.value = true;
      } catch (e) { toast(e.message, "error"); }
    }
    async function applySplit() {
      try {
        const r = await api.breakup(props.id, true);
        toast(`Split into ${r.new_persons.length} persons`, "success");
        showSplit.value = false;
        await refreshPersons();
        await load();
      } catch (e) { toast(e.message, "error"); }
    }
    function openReview() {
      reviewTarget.value = props.id;
    }

    const groups = computed(() => groupByDay(photos.value));
    const viewerItems = computed(() => photos.value.map((p) => ({ ...p })));

    watch(() => props.id, load);
    onMounted(load);

    const router = VueRouter.useRouter();

    return {
      person, loading, err, photos, faces, groups, viewerItems, viewerStart,
      renaming, newName, showSplit, splitGroups, reviewTarget,
      startRename, saveRename, togglePin, doReset, previewSplit, applySplit,
      openReview, personLabel, router,
    };
  },
  template: `
  <div class="view">
    <div v-if="loading" class="empty">Loading…</div>
    <div v-else-if="err" class="empty err">{{ err }}</div>
    <div v-else-if="person">
      <div class="person-head">
        <button class="btn ghost" type="button" @click="router.push('/people')">‹ People</button>
        <div class="ph-avatar">
          <img v-if="person.avatar" :src="person.avatar" alt="">
          <div v-else class="pc-avatar-empty">?</div>
        </div>
        <div class="ph-id">
          <template v-if="renaming">
            <input class="input" v-model="newName" placeholder="Person name"
                   @keyup.enter="saveRename">
            <button class="btn primary" type="button" @click="saveRename">Save</button>
            <button class="btn ghost" type="button" @click="renaming = false">Cancel</button>
          </template>
          <template v-else>
            <h1 class="ph-label">
              {{ personLabel(person) }}
              <span class="chip" v-if="person.seq != null" :title="'DB id ' + person.id">id {{ person.id }}</span>
              <span class="chip pin" v-if="person.pinned">pinned</span>
            </h1>
            <div class="ph-actions">
              <button class="btn ghost" type="button" @click="startRename">Rename</button>
              <button class="btn ghost" type="button" @click="togglePin">
                {{ person.pinned ? 'Unpin' : 'Pin' }}
              </button>
              <button class="btn ghost" type="button" @click="openReview">Review</button>
              <button class="btn ghost" type="button" @click="previewSplit">Mixed grouping?</button>
              <button class="btn danger" type="button" @click="doReset($event.target)">Reset</button>
            </div>
          </template>
          <div class="ph-meta">{{ person.total }} photos · faces {{ faces.length }}</div>
        </div>
      </div>

      <!-- split preview -->
      <div class="split-panel" v-if="showSplit">
        <div class="split-head">
          <strong>Split into {{ splitGroups.length }} groups?</strong>
          <div class="split-actions">
            <button class="btn primary" type="button" @click="applySplit">Apply split</button>
            <button class="btn ghost" type="button" @click="showSplit = false">Cancel</button>
          </div>
        </div>
        <div class="split-groups">
          <div class="split-group" v-for="(g, i) in splitGroups" :key="i">
            <div class="sg-label">Group {{ i + 1 }} · {{ g.length }}</div>
            <div class="sg-thumbs">
              <img v-for="f in g" :key="f.face_id" :src="f.thumb" alt="">
            </div>
          </div>
        </div>
      </div>

      <!-- per-person review -->
      <div class="person-review" v-if="reviewTarget != null">
        <div class="pr-head">
          <strong>Review suggestions</strong>
          <div class="split-actions">
            <button class="btn ghost" type="button" @click="reviewTarget = null">Close</button>
            <router-link class="btn ghost" :to="'/review?t=' + reviewTarget">Full queue</router-link>
          </div>
        </div>
        <ReviewCard :target-id="reviewTarget" />
      </div>

      <div class="view-head sub"><h2>Photos <span class="count">{{ person.total }}</span></h2></div>
      <div class="photo-groups" v-for="g in groups" :key="g.day">
        <div class="day-label">{{ g.day }}</div>
        <div class="photo-grid">
          <button class="photo-cell" v-for="p in g.items" :key="p.id"
                  type="button" @click="viewerStart = photos.indexOf(p)">
            <img :src="p.thumb" loading="lazy" alt="">
            <span class="reason" v-if="p.reason && p.reason !== 'face'">{{ p.reason }}</span>
          </button>
        </div>
      </div>

      <div class="face-strip" v-if="faces.length">
        <div class="fs-label">Clustered faces</div>
        <div class="fs-scroll">
          <img v-for="f in faces" :key="f.id" :src="f.thumb" loading="lazy" alt="">
        </div>
      </div>

      <PhotoViewer v-if="viewerStart >= 0" :items="viewerItems"
                   :start="viewerStart" @close="viewerStart = -1" />
    </div>
  </div>
  `,
};
