/* Photos view: date-grouped grid of all photos. */

const { ref, onMounted, computed } = Vue;
import { api } from "./api.js";
import { store, refreshStatus } from "./store.js";
import { groupByDay, toast } from "./ui.js";
import { PhotoViewer } from "./viewer.js";

export const PhotosView = {
  name: "PhotosView",
  components: { PhotoViewer },
  setup() {
    const photos = ref([]);
    const total = ref(0);
    const loading = ref(false);
    const allLoaded = ref(false);
    const viewerStart = ref(-1);

    async function load(more = false) {
      if (loading.value) return;
      loading.value = true;
      try {
        const offset = more ? photos.value.length : 0;
        const r = await api.photos(offset, 60);
        total.value = r.total;
        if (more) photos.value = photos.value.concat(r.photos);
        else photos.value = r.photos;
        allLoaded.value = photos.value.length >= r.total;
      } catch (e) {
        toast(e.message, "error");
      } finally {
        loading.value = false;
      }
    }

    function loadMore() {
      if (!allLoaded.value) load(true);
    }

    const groups = computed(() => groupByDay(photos.value));
    const viewerItems = computed(() => photos.value.map((p) => ({ ...p })));

    onMounted(() => {
      load();
      refreshStatus();
    });

    return { store, photos, total, loading, allLoaded, groups,
             viewerItems, viewerStart, load, loadMore };
  },
  template: `
  <div class="view">
    <div class="view-head">
      <h1>Photos <span class="count" v-if="total">{{ total }}</span></h1>
      <div class="view-head-side">
        <span class="muted" v-if="store.status.pending">
          {{ store.status.pending }} processing…
        </span>
      </div>
    </div>

    <div v-if="!photos.length && !loading" class="empty">
      <h2>No photos yet</h2>
      <p>Add photos from Settings, or scan a folder on the server.</p>
      <router-link class="btn primary" to="/settings">Open Settings</router-link>
    </div>

    <div class="photo-groups" v-for="g in groups" :key="g.day">
      <div class="day-label">{{ g.day }}</div>
      <div class="photo-grid">
        <button class="photo-cell" v-for="p in g.items" :key="p.id"
                type="button" @click="viewerStart = photos.indexOf(p)">
          <img :src="p.thumb" loading="lazy" alt="">
        </button>
      </div>
    </div>

    <div class="more" v-if="!allLoaded && photos.length">
      <button class="btn ghost" type="button" :disabled="loading" @click="loadMore">
        {{ loading ? 'Loading…' : 'Load more' }}
      </button>
    </div>

    <PhotoViewer v-if="viewerStart >= 0" :items="viewerItems"
                 :start="viewerStart" @close="viewerStart = -1" />
  </div>
  `,
};
