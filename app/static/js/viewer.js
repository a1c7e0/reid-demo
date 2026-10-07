/* Photo detail viewer: the image with face/body box overlays, an ML info
 * panel (per-face and per-body facts + Tom Select person reassignment), and
 * prev/next navigation. PhotoSwipe is used as the wheel for fullscreen zoom. */

const { ref, computed, watch, onMounted, onUnmounted, nextTick } = Vue;
import { api, personLabel, personHref } from "./api.js";
import { toast } from "./ui.js";

/* Tom Select person picker: avatar preview + similarity-desc candidates.
 * kind: "face" | "det" | "person"; on pick it POSTs the label. */
const PersonPicker = {
  name: "PersonPicker",
  props: {
    kind: { type: String, required: true },
    refId: { type: Number, required: true },
    currentId: { type: Number, default: null },
  },
  emits: ["changed"],
  setup(props, { emit }) {
    const el = ref(null);
    let ts = null;
    let destroyed = false;

    async function build() {
      let items = [];
      try {
        const r = await api.suggest(props.kind, props.refId);
        items = r.items || [];
      } catch (e) {
        toast(e.message, "error");
      }
      if (destroyed || !el.value) return;

      const options = [{ id: "__new", label: "+ New person", thumb: null }];
      const current = props.currentId;
      for (const it of items) {
        options.push({
          id: String(it.person_id),
          label: it.display,
          thumb: it.thumb,
          sim: it.sim,
          photos: it.photos,
          isCurrent: it.person_id === current,
        });
      }
      if (current) {
        options.push({ id: "__none", label: "Unassign", thumb: null });
      }

      ts = new TomSelect(el.value, {
        options,
        items: current ? [String(current)] : [],
        allowEmptyOption: true,
        maxItems: 1,
        labelField: "label",
        valueField: "id",
        searchField: ["label"],
        placeholder: "No person",
        render: {
          option: (data, escape) => {
            const av = data.thumb
              ? `<img class="ts-av" src="${data.thumb}" alt="">`
              : `<span class="ts-av ts-av-empty"></span>`;
            const sim =
              data.sim != null
                ? `<span class="ts-sim">${Math.round(data.sim * 100)}%</span>`
                : "";
            return `<div class="ts-row">${av}<span class="ts-name">${escape(
              data.label
            )}</span>${sim}</div>`;
          },
          item: (data, escape) => {
            const av = data.thumb
              ? `<img class="ts-av" src="${data.thumb}" alt="">`
              : `<span class="ts-av ts-av-empty"></span>`;
            return `<div class="ts-row">${av}<span class="ts-name">${escape(
              data.label
            )}</span></div>`;
          },
        },
        onChange: async (value) => {
          if (!value) return;
          let action, personId;
          if (value === "__new") {
            action = "new";
          } else if (value === "__none") {
            action = "unassign";
          } else {
            action = "assign";
            personId = parseInt(value, 10);
          }
          try {
            const r = await api.label(props.kind, props.refId, {
              action,
              person_id: personId,
            });
            if (action === "new" && r.person_id) ts.setValue(String(r.person_id));
            toast(`Labelled as ${action}`, "success");
            emit("changed", r.person_id);
          } catch (e) {
            toast(e.message, "error");
          }
        },
      });
    }

    onMounted(build);
    watch(
      () => [props.kind, props.refId],
      () => {
        if (ts) {
          ts.destroy();
          ts = null;
        }
        nextTick(build);
      }
    );
    return { el };
  },
  template: `<select ref="el"></select>`,
};

/* Fullscreen zoom of a single image via PhotoSwipe (the wheel). */
async function openZoom(src, w, h) {
  try {
    const mod = await import("/vendor/photoswipe.esm.min.js");
    const PhotoSwipe = mod.default;
    const ps = new PhotoSwipe({
      dataSource: [{ src, w, h, msrc: src }],
      index: 0,
      showHideAnimationType: "fade",
      bgOpacity: 0.9,
    });
    ps.init();
  } catch (e) {
    toast("zoom failed: " + e.message, "error");
  }
}

export const PhotoViewer = {
  name: "PhotoViewer",
  components: { PersonPicker },
  props: {
    items: { type: Array, required: true },  // [{id, thumb, ...}]
    start: { type: Number, default: 0 },
  },
  emits: ["close"],
  setup(props, { emit }) {
    const idx = ref(props.start);
    const detail = ref(null);
    const loading = ref(false);
    const err = ref(null);
    const showFaces = ref(true);
    const showDets = ref(true);
    const showInfo = ref(true);
    const img = ref(null);
    const scale = ref(1);

    const current = computed(() => props.items[idx.value] || null);

    async function load() {
      const it = props.items[idx.value];
      if (!it) return;
      loading.value = true;
      err.value = null;
      try {
        detail.value = await api.photo(it.id);
        await nextTick();
        measure();
      } catch (e) {
        err.value = e.message;
      } finally {
        loading.value = false;
      }
    }

    function measure() {
      const i = img.value;
      if (i && i.naturalWidth) {
        scale.value = i.clientWidth / i.naturalWidth;
      }
    }

    /* Boxes are stored in original-image pixels; the layer is laid over the
     * rendered image, so percentages keep them aligned no matter how the
     * image is scaled or how late it finishes loading (the old pixel math
     * depended on measuring the <img>, which missed cached loads and put
     * every box outside the picture). */
    function boxStyle(box) {
      if (!box || !detail.value) return {};
      const w = detail.value.width || 1, h = detail.value.height || 1;
      return {
        left: (box[0] / w) * 100 + "%",
        top: (box[1] / h) * 100 + "%",
        width: ((box[2] - box[0]) / w) * 100 + "%",
        height: ((box[3] - box[1]) / h) * 100 + "%",
      };
    }

    function go(delta) {
      const n = idx.value + delta;
      if (n >= 0 && n < props.items.length) {
        idx.value = n;
      }
    }

    function onKey(e) {
      if (e.key === "Escape") emit("close");
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === "ArrowRight") go(1);
    }

    watch(idx, load);
    onMounted(() => {
      load();
      window.addEventListener("keydown", onKey);
    });
    const cleanup = () => window.removeEventListener("keydown", onKey);
    onUnmounted(cleanup);

    function zoom() {
      if (detail.value) {
        openZoom(detail.value.original, detail.value.width, detail.value.height);
      }
    }

    function fmtTime(t) {
      try {
        return new Date(t * 1000).toLocaleString(undefined, {
          year: "numeric", month: "short", day: "numeric",
          hour: "2-digit", minute: "2-digit",
        });
      } catch (e) { return String(t); }
    }
    function fmtBytes(n) {
      if (n == null) return "";
      const u = ["B", "KB", "MB", "GB"];
      let i = 0, v = n;
      while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
      return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${u[i]}`;
    }
    function personsFor(d) {
      return d.person;
    }

    function onReassigned() {
      load();
    }

    return {
      idx, detail, loading, err, showFaces, showDets, showInfo,
      fmtTime, fmtBytes,
      img, current, scale, boxStyle, go, zoom, onReassigned,
      personLabel, personHref, pctFmt: (x) => (x == null ? "-" : Math.round(x * 100) + "%"),
      nOf: () => (props.items.length ? idx.value + 1 : 0),
    };
  },
  template: `
  <div class="viewer" @click.self="$emit('close')">
    <div class="viewer-top">
      <div class="viewer-count">{{ nOf() }} / {{ items.length }}</div>
      <div class="viewer-actions">
        <button class="btn ghost" type="button" @click="showInfo = !showInfo">
          {{ showInfo ? 'Hide info' : 'Show info' }}
        </button>
        <button class="btn ghost" type="button" @click="zoom">Zoom</button>
        <button class="btn ghost" type="button" @click="$emit('close')">Close</button>
      </div>
    </div>

    <div class="viewer-legend">
      <span class="lg"><i class="sw face"></i>face box</span>
      <span class="lg"><i class="sw det"></i>body box</span>
      <span class="lg"><i class="sw asn"></i>assigned to a person</span>
      <span class="lg muted">tap a row in Info to rebind</span>
    </div>

    <div class="viewer-body" :class="{ 'with-info': showInfo }">
      <div class="viewer-stage" @click="go(1)">
        <button class="nav prev" type="button" @click.stop="go(-1)" aria-label="Previous">‹</button>
        <div class="img-wrap" v-if="detail">
          <img ref="img" :src="detail.original" :alt="''" @load="measure" />
          <div class="box-layer">
            <template v-if="showDets">
              <div v-for="d in detail.detections" :key="'d'+d.id"
                   class="box det" :class="{ assigned: d.person && d.person.id }"
                   :style="boxStyle(d.box)">
                <span class="box-tag">
                  <b v-if="d.person && d.person.id">{{ d.person.display }}</b>
                  <b v-else class="none">body</b>
                  <em v-if="d.reason">{{ d.reason }}</em>
                </span>
              </div>
            </template>
            <template v-if="showFaces">
              <div v-for="f in detail.faces" :key="'f'+f.id"
                   class="box face" :class="{ assigned: f.person && f.person.id }"
                   :style="boxStyle(f.box)">
                <span class="box-tag">
                  <b v-if="f.person && f.person.id">{{ f.person.display }}</b>
                  <b v-else class="none">face</b>
                  <em v-if="f.sideways">sideways</em>
                </span>
              </div>
            </template>
          </div>
        </div>
        <div class="viewer-empty" v-else-if="loading">Loading…</div>
        <div class="viewer-empty err" v-else-if="err">{{ err }}</div>
        <button class="nav next" type="button" @click.stop="go(1)" aria-label="Next">›</button>
      </div>

      <aside class="viewer-info" v-if="showInfo && detail">
        <div class="info-toggles">
          <label class="chk"><input type="checkbox" v-model="showFaces"> Faces</label>
          <label class="chk"><input type="checkbox" v-model="showDets"> Bodies</label>
        </div>

        <div class="info-block">
          <h3>Photo</h3>
          <div class="ml-facts meta-facts">
            <span v-if="detail.taken_at">{{ fmtTime(detail.taken_at) }}</span>
            <span>{{ detail.width }}×{{ detail.height }}</span>
            <span v-if="detail.bytes">{{ fmtBytes(detail.bytes) }}</span>
            <span class="mono" :title="detail.path">{{ detail.path }}</span>
            <span class="mono" v-if="detail.sha1">sha1 {{ detail.sha1.slice(0, 10) }}</span>
          </div>
        </div>

        <div class="info-block" v-if="detail.faces && detail.faces.length">
          <h3>Faces ({{ detail.faces.length }})</h3>
          <div class="ml-row" v-for="f in detail.faces" :key="f.id">
            <img class="ml-thumb" :src="'/face_thumb/' + f.id + '.jpg'" alt="">
            <div class="ml-meta">
              <div class="ml-facts">
                <span>score {{ f.score?.toFixed?.(2) ?? '-' }}</span>
                <span>blur {{ f.blur ?? '-' }}</span>
                <span>yaw {{ f.yaw != null ? f.yaw.toFixed(2) : '-' }}</span>
                <span v-if="f.sideways" class="warn">sideways</span>
              </div>
              <PersonPicker kind="face" :ref-id="f.id"
                            :current-id="f.person ? f.person.id : null"
                            @changed="onReassigned" />
            </div>
          </div>
        </div>

        <div class="info-block" v-if="detail.detections && detail.detections.length">
          <h3>Bodies ({{ detail.detections.length }})</h3>
          <div class="ml-row" v-for="d in detail.detections" :key="d.id">
            <img class="ml-thumb" :src="'/det_thumb/' + d.id + '.jpg'" alt="">
            <div class="ml-meta">
              <div class="ml-facts">
                <span>conf {{ d.conf?.toFixed?.(2) ?? '-' }}</span>
                <span v-if="d.reason">reason {{ d.reason }}</span>
                <span v-if="d.personness != null">pness {{ d.personness.toFixed(2) }}</span>
                <span v-if="d.margin != null">margin {{ d.margin.toFixed(2) }}</span>
              </div>
              <PersonPicker kind="det" :ref-id="d.id"
                            :current-id="d.person ? d.person.id : null"
                            @changed="onReassigned" />
            </div>
          </div>
        </div>

        <div class="info-empty" v-if="!detail.faces.length && !detail.detections.length">
          Nothing detected in this photo.
        </div>
      </aside>
    </div>
  </div>
  `,
};
