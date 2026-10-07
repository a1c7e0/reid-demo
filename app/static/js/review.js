/* Review: Ente-style suggestions shown ONE AT A TIME.
 *
 * Left  = target person (the anchor, single head image)
 * Right = candidate cluster (its member photos)
 * Evidence row = similarity / median / best / body / time gap.
 * Keyboard: Enter merge, Backspace keep-separate, S skip, U undo. */

const { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } = Vue;
import { api, personLabel, personHref } from "./api.js";
import { refreshPersons } from "./store.js";
import { toast } from "./ui.js";

export const ReviewCard = {
  name: "ReviewCard",
  props: {
    targetId: { type: Number, default: null },
  },
  setup(props) {
    const items = ref([]);
    const remaining = ref(0);
    const idx = ref(0);
    const loading = ref(true);
    const busy = ref(false);

    async function load() {
      loading.value = true;
      try {
        const r = await api.review(40, props.targetId || undefined);
        items.value = r.items || [];
        remaining.value = r.remaining || 0;
        idx.value = 0;
      } catch (e) {
        toast(e.message, "error");
      } finally {
        loading.value = false;
      }
    }

    const item = computed(() => items.value[idx.value] || null);
    const candidateThumbs = computed(() => {
      const it = item.value;
      if (!it) return [];
      return (it.candidate.thumbs || []).slice(0, 4);
    });

    async function act(kind) {
      const it = item.value;
      if (!it || busy.value) return;
      busy.value = true;
      const body = {
        target_id: it.target_id,
        candidate_id: it.candidate_id,
        det_ids: it.det_ids,
      };
      try {
        if (kind === "merge") {
          await api.reviewMerge(body);
          toast(`Merged into ${it.target.display}`, "success");
        } else if (kind === "reject") {
          await api.reviewReject(body);
          toast("Kept separate", "info");
        } else if (kind === "skip") {
          await api.reviewSkip(body);
          toast("Skipped", "info");
        } else if (kind === "undo") {
          await api.reviewUndo();
          toast("Undone", "success");
        }
        await refreshPersons();
        await load();
      } catch (e) {
        toast(e.message, "error");
      } finally {
        busy.value = false;
      }
    }

    function onKey(e) {
      if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
      if (!item.value && e.key !== "u" && e.key !== "U") return;
      if (e.key === "Enter") { e.preventDefault(); act("merge"); }
      else if (e.key === "Backspace") { e.preventDefault(); act("reject"); }
      else if (e.key === "s" || e.key === "S") { e.preventDefault(); act("skip"); }
      else if (e.key === "u" || e.key === "U") { e.preventDefault(); act("undo"); }
    }

    onMounted(() => {
      load();
      window.addEventListener("keydown", onKey);
    });
    onBeforeUnmount(() => window.removeEventListener("keydown", onKey));
    watch(() => props.targetId, load);

    return { items, item, remaining, idx, loading, busy, candidateThumbs,
             act, personLabel };
  },
  template: `
  <div class="review-card">
    <div class="rc-status" v-if="loading">Loading suggestions…</div>
    <div class="rc-status" v-else-if="!item">
      Nothing left to review. <span class="muted">{{ remaining }} total in queue.</span>
    </div>

    <template v-else>
      <div class="rc-top">
        <div class="rc-counter">{{ remaining }} left</div>
        <div class="rc-evidence">
          <span class="ev" title="overall score">{{ Math.round(item.sim * 100) }}%</span>
          <span class="ev-sub" v-if="item.sim_med != null">med {{ Math.round(item.sim_med * 100) }}%</span>
          <span class="ev-sub" v-if="item.sim_best != null">best {{ Math.round(item.sim_best * 100) }}%</span>
          <span class="ev-sub" v-if="item.body_best != null">body {{ Math.round(item.body_best * 100) }}%</span>
          <span class="ev-sub" v-if="item.gap_days != null">{{ item.gap_days }}d apart</span>
        </div>
      </div>

      <div class="rc-pair">
        <div class="rc-side target">
          <div class="rc-side-label">Merge into this person</div>
          <img class="rc-thumb big" v-if="item.target.thumb" :src="item.target.thumb" alt="">
          <div class="rc-name">{{ item.target.display }}</div>
          <div class="rc-count">{{ item.target.photos }} photos</div>
        </div>

        <div class="rc-arrow">→</div>

        <div class="rc-side candidate">
          <div class="rc-side-label">Candidate · keep or merge</div>
          <div class="rc-thumbs">
            <img v-for="t in candidateThumbs" :key="t.photo_id" :src="t.url" alt="">
          </div>
          <div class="rc-name">{{ item.candidate.display }}</div>
          <div class="rc-count">{{ item.candidate.photos }} photos</div>
        </div>
      </div>

      <div class="rc-actions">
        <button class="btn primary" type="button" :disabled="busy" @click="act('merge')">
          Merge <span class="kbd">Enter</span>
        </button>
        <button class="btn ghost" type="button" :disabled="busy" @click="act('reject')">
          Keep separate <span class="kbd">⌫</span>
        </button>
        <button class="btn ghost" type="button" :disabled="busy" @click="act('skip')">
          Skip <span class="kbd">S</span>
        </button>
        <button class="btn ghost" type="button" :disabled="busy" @click="act('undo')">
          Undo <span class="kbd">U</span>
        </button>
      </div>
    </template>
  </div>
  `,
};

/* Review page: queue + per-person scope + decided history. */
export const ReviewView = {
  name: "ReviewView",
  components: { ReviewCard },
  setup() {
    const targetId = ref(null);
    const targetLabel = ref("");
    const decided = ref([]);
    const showDecided = ref(false);

    // the scope chip must show the person's label (name / person #seq),
    // never the raw DB id - ids are rowid values and mean nothing in the UI
    async function loadTargetLabel() {
      targetLabel.value = "";
      if (!targetId.value) return;
      try {
        const p = await api.person(targetId.value);
        targetLabel.value = personLabel(p) || `person #${targetId.value}`;
      } catch (e) {
        targetLabel.value = `person #${targetId.value}`;
      }
    }
    watch(targetId, loadTargetLabel);

    async function loadDecided() {
      try {
        const r = await api.reviewDecided();
        decided.value = r.items || [];
      } catch (e) { toast(e.message, "error"); }
    }
    async function reopen(a, b) {
      try {
        await api.reviewReopen(a, b);
        toast("Reopened", "success");
        await loadDecided();
      } catch (e) { toast(e.message, "error"); }
    }

    onMounted(() => {
      const q = new URLSearchParams((location.hash.split("?")[1] || ""));
      const t = q.get("t");
      if (t) targetId.value = parseInt(t, 10);
      loadDecided();
    });

    return { targetId, targetLabel, decided, showDecided, loadDecided, reopen };
  },
  template: `
  <div class="view">
    <div class="view-head">
      <h1>Review</h1>
      <div class="view-head-side">
        <button class="btn ghost" type="button" @click="showDecided = !showDecided">
          {{ showDecided ? 'Hide decided' : 'Decided history' }}
        </button>
        <button class="btn ghost" type="button" v-if="targetId" @click="targetId = null">
          Clear filter
        </button>
      </div>
    </div>

    <div class="scope" v-if="targetId">Scoped to {{ targetLabel }}</div>

    <ReviewCard :target-id="targetId" />

    <div class="decided" v-if="showDecided">
      <h2>Decided <span class="count">{{ decided.length }}</span></h2>
      <div class="decided-list">
        <div class="dec-row" v-for="(d, i) in decided" :key="i">
          <img class="d-thumb" v-if="d.a_thumb" :src="d.a_thumb" alt="">
          <div class="d-names">
            <div>{{ d.a_display }}</div>
            <div>{{ d.b_display }}</div>
          </div>
          <img class="d-thumb" v-if="d.b_thumb" :src="d.b_thumb" alt="">
          <span class="d-status" :class="d.status">{{ d.status }}</span>
          <button v-if="d.status === 'rejected'" class="btn ghost sm"
                  type="button" @click="reopen(d.a, d.b)">Reopen</button>
        </div>
        <div v-if="!decided.length" class="muted">No decisions yet.</div>
      </div>
    </div>
  </div>
  `,
};
