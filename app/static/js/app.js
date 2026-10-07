/* App shell, modelled on Ente's mobile app:
 *   - floating bottom pill tab bar (4 tabs), tap current tab = scroll to top
 *   - horizontal swipe on the content switches tabs
 *   - the tab bar steps aside whenever a selection is active and a
 *     selection action bar takes its place
 *   - a slim top app bar per screen (title + search), not a desktop header
 */

const { createApp, ref, computed, onMounted, onBeforeUnmount } = Vue;
const { createRouter, createWebHashHistory, RouterView, RouterLink, useRoute } = VueRouter;

import { api, personLabel, personHref } from "./api.js";
import { store, selection, refreshStatus, refreshPersons } from "./store.js";
import { toast } from "./ui.js";
import { PhotosView } from "./photos.js";
import { PeopleView } from "./people.js";
import { PersonView } from "./person.js";
import { ReviewView } from "./review.js";
import { SettingsView } from "./settings.js";

const TABS = [
  { path: "/photos", label: "Photos" },
  { path: "/people", label: "People" },
  { path: "/review", label: "Review" },
  { path: "/settings", label: "Settings" },
];

const App = {
  name: "App",
  components: { RouterView, RouterLink },
  setup() {
    const search = ref("");
    const searchOpen = ref(false);
    const route = useRoute();
    const router = VueRouter.useRouter();

    const results = computed(() => {
      const q = search.value.trim().toLowerCase();
      if (!q) return [];
      return store.persons
        .filter((p) =>
          (p.name || "").toLowerCase().includes(q) ||
          (p.display || "").toLowerCase().includes(q))
        .slice(0, 8);
    });

    function pick(p) {
      search.value = "";
      searchOpen.value = false;
      router.push(personHref(p));
    }

    const activeTab = computed(() => {
      const p = route.path;
      if (p.startsWith("/person/")) return "/people";
      return TABS.find((t) => p.startsWith(t.path))?.path || "/photos";
    });

    function onTab(path) {
      if (path === activeTab.value) {
        // Ente: tapping the tab you are on scrolls the screen back to the top
        window.scrollTo({ top: 0, behavior: "smooth" });
        return;
      }
      router.push(path);
    }

    /* Ente: the tabs live in a horizontal pager, so a swipe moves between
     * them. Deliberately narrow in what it accepts - a swipe must start and
     * end as a mostly-horizontal drag without pinching. */
    let start = null;
    function onPointerDown(e) {
      if (e.pointerType === "mouse" && e.button !== 0) return;
      start = { x: e.clientX, y: e.clientY, t: Date.now() };
    }
    function onPointerUp(e) {
      if (!start) return;
      const dx = e.clientX - start.x;
      const dy = e.clientY - start.y;
      const dt = Date.now() - start.t;
      start = null;
      if (selection.active) return;
      if (dt > 700 || Math.abs(dx) < 64 || Math.abs(dy) > Math.abs(dx) * 0.6) return;
      // ignore swipes that started on a horizontally scrollable element
      if (e.target.closest?.(".no-tab-swipe")) return;
      const i = TABS.findIndex((t) => t.path === activeTab.value);
      const next = dx < 0 ? i + 1 : i - 1;
      if (next >= 0 && next < TABS.length) router.push(TABS[next].path);
    }

    onMounted(async () => {
      await Promise.all([refreshStatus(), refreshPersons()]);
      window.addEventListener("pointerdown", onPointerDown, { passive: true });
      window.addEventListener("pointerup", onPointerUp, { passive: true });
      setInterval(async () => {
        if (store.status.pending > 0 || store.status.worker.queue > 0) {
          await refreshStatus();
        }
      }, 4000);
    });
    onBeforeUnmount(() => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
    });

    return { store, selection, search, searchOpen, results, pick,
             personLabel, activeTab, onTab, TABS };
  },
  template: `
  <div class="app">
    <header class="app-bar">
      <div class="app-bar-title">{{ TABS.find(t => t.path === activeTab)?.label }}</div>
      <div class="app-bar-actions">
        <div class="search">
          <button class="icon-btn" type="button" @click="searchOpen = !searchOpen"
                  aria-label="Search people">
            <svg viewBox="0 0 24 24" width="20" height="20"><g fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round">
               <circle cx="11" cy="11" r="7"/><path d="M20 20l-3.2-3.2"/></g></svg>
          </button>
          <div class="search-results" v-if="searchOpen && results.length">
            <button class="sr-row" type="button" v-for="p in results" :key="p.id"
                    @mousedown.prevent="pick(p)">
              <img v-if="p.cover" :src="p.cover" alt="">
              <span class="sr-name">{{ personLabel(p) }}</span>
              <span class="sr-count">{{ p.photos }}</span>
            </button>
          </div>
        </div>
      </div>
    </header>

    <div class="search-sheet" v-if="searchOpen">
      <input class="input" type="search" v-model="search" placeholder="Search people"
             autofocus>
    </div>

    <main class="main">
      <router-view />
    </main>

    <nav class="tab-bar" :class="{ hidden: selection.active }" aria-label="Sections">
      <button class="tab" v-for="t in TABS" :key="t.path" type="button"
              :class="{ active: activeTab === t.path }" @click="onTab(t.path)">
        <span class="tab-label">{{ t.label }}</span>
      </button>
    </nav>
  </div>
  `,
};

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", redirect: "/photos" },
    { path: "/photos", component: PhotosView },
    { path: "/people", component: PeopleView },
    { path: "/person/:id", component: PersonView,
      props: (route) => ({ id: parseInt(route.params.id, 10) }) },
    { path: "/review", component: ReviewView },
    { path: "/settings", component: SettingsView },
    { path: "/:pathMatch(.*)*", redirect: "/photos" },
  ],
});

createApp(App).use(router).mount("#app");
