/* Settings: status, scan, upload, re-cluster, reset options, label export. */

const { ref, onMounted } = Vue;
import { api } from "./api.js";
import { store, refreshStatus, refreshPersons } from "./store.js";
import { toast, armConfirm } from "./ui.js";

const RESETS = [
  ["review", "Reset review history (decisions + undo log)?"],
  ["labels", "Reset ALL labels?"],
  ["manual", "Reset ALL manual marks (labels + review + merge pins)?"],
  ["full", "FULL re-cluster: wipes persons, names, labels and re-groups from scratch. Continue?"],
];

export const SettingsView = {
  name: "SettingsView",
  setup() {
    const cfg = ref(null);
    onMounted(async () => {
      try { cfg.value = await api.config(); } catch (e) { /* non-fatal */ }
    });
    const scanPath = ref("");
    const scanBusy = ref(false);
    const reclusterBusy = ref(false);
    const uploadBusy = ref(false);
    const statusMsg = ref("");
    const fileInput = ref(null);

    async function doScan() {
      if (!scanPath.value.trim()) { toast("Enter a folder path", "warn"); return; }
      scanBusy.value = true;
      try {
        const r = await api.scan(scanPath.value.trim());
        toast(`Scan: imported ${r.imported ?? 0}, found ${r.found ?? 0}`, "success");
        await refreshStatus();
      } catch (e) { toast(e.message, "error"); }
      finally { scanBusy.value = false; }
    }

    async function doRecluster(btn) {
      const ok = await armConfirm(btn);
      if (!ok) return;
      reclusterBusy.value = true;
      try {
        const r = await api.recluster();
        toast("Re-clustered", "success");
        await refreshStatus();
        await refreshPersons();
      } catch (e) { toast(e.message, "error"); }
      finally { reclusterBusy.value = false; }
    }

    async function doReset(mode, btn) {
      const ok = await armConfirm(btn);
      if (!ok) return;
      statusMsg.value = "working…";
      try {
        const r = await api.adminReset(mode);
        statusMsg.value = "done: " + JSON.stringify(r);
        toast("Reset done", "success");
        await refreshStatus();
        await refreshPersons();
      } catch (e) {
        statusMsg.value = "failed: " + e.message;
        toast(e.message, "error");
      }
    }

    async function doUpload(e) {
      const files = e.target.files;
      if (!files || !files.length) return;
      uploadBusy.value = true;
      try {
        const r = await api.upload(Array.from(files));
        toast(`Uploaded ${r.results.length} file(s)`, "success");
        await refreshStatus();
      } catch (err) { toast(err.message, "error"); }
      finally {
        uploadBusy.value = false;
        if (fileInput.value) fileInput.value.value = "";
      }
    }

    async function refreshNow() {
      await refreshStatus();
      toast("Status refreshed", "info", 1400);
    }

    onMounted(refreshStatus);
    return { store, scanPath, scanBusy, reclusterBusy, uploadBusy,
             statusMsg, fileInput, RESETS, doScan, doRecluster, doReset,
             cfg, doUpload, refreshNow, labelsHref: api.labelsExport };
  },
  template: `
  <div class="view">
    <div class="view-head"><h1>Settings</h1></div>

    <section class="card">
      <h2>Status</h2>
      <div class="stat-grid">
        <div class="stat"><b>{{ store.status.photos }}</b><span>photos</span></div>
        <div class="stat"><b>{{ store.status.processed }}</b><span>processed</span></div>
        <div class="stat"><b>{{ store.status.pending }}</b><span>pending</span></div>
        <div class="stat"><b>{{ store.status.persons }}</b><span>people</span></div>
        <div class="stat"><b>{{ store.status.faces }}</b><span>faces</span></div>
        <div class="stat"><b>{{ store.status.detections }}</b><span>bodies</span></div>
      </div>
      <div class="worker muted">
        worker: queue {{ store.status.worker.queue }},
        processed {{ store.status.worker.processed }},
        failed {{ store.status.worker.failed }}
        <span v-if="store.status.worker.last_error" class="err">
          — {{ store.status.worker.last_error }}
        </span>
      </div>
      <button class="btn ghost" type="button" @click="refreshNow">Refresh</button>
    </section>

    <section class="card" v-if="cfg">
      <h2>Model &amp; configuration</h2>
      <div class="kv-grid">
        <div class="kv"><span>Face detector</span><b>{{ cfg.models.face_detector.file }}
          <em>{{ cfg.models.face_detector.kind }}</em></b></div>
        <div class="kv"><span>Face embedder</span><b>{{ cfg.models.face_embedder.file }}
          <em>{{ cfg.models.face_embedder.preset }}, {{ cfg.models.face_embedder.size_mb }} MB</em></b></div>
        <div class="kv"><span>Person detector</span><b>{{ cfg.models.person_detector.file }}
          <em>{{ cfg.models.person_detector.size_mb }} MB</em></b></div>
        <div class="kv"><span>Body embedder</span><b>{{ cfg.models.body_embedder.file }}
          <em>{{ cfg.models.body_embedder.size_mb }} MB</em></b></div>
        <div class="kv"><span>Personness gate</span><b>{{ cfg.models.pose_gate.file }}
          <em>{{ cfg.models.pose_gate.size_mb }} MB</em></b></div>
      </div>

      <h3>Clustering ({{ cfg.engine }})</h3>
      <div class="kv-grid">
        <div class="kv"><span>join bar, good neighbour</span><b>{{ cfg.clustering.t_good }}</b></div>
        <div class="kv"><span>join bar, bad neighbour</span><b>{{ cfg.clustering.t_bad }}</b></div>
        <div class="kv"><span>auto-eject bar</span><b>{{ cfg.clustering.eject_bar || 'off' }}</b></div>
        <div class="kv"><span>one person per photo</span><b>{{ cfg.clustering.one_per_photo ? 'yes' : 'no' }}</b></div>
      </div>

      <h3>Body attach (back views)</h3>
      <div class="kv-grid">
        <div class="kv"><span>enabled</span><b>{{ cfg.body_attach.enabled ? 'yes' : 'no' }}</b></div>
        <div class="kv"><span>join bar</span><b>{{ cfg.body_attach.join_t }}</b></div>
        <div class="kv"><span>ambiguity margin</span><b>{{ cfg.body_attach.margin }}</b></div>
        <div class="kv"><span>capture-time window</span><b>{{ cfg.body_attach.max_gap_s / 3600 }} h</b></div>
      </div>

      <h3>Review &amp; gates</h3>
      <div class="kv-grid">
        <div class="kv"><span>review band (face)</span><b>≥ {{ cfg.review.sim_min }}</b></div>
        <div class="kv"><span>review band (body)</span><b>≥ {{ cfg.review.body_sim_min }}</b></div>
        <div class="kv"><span>review order</span><b>MMR λ={{ cfg.review.mmr_lambda }}
          <em>{{ cfg.review.mmr_impact ? 'impact-weighted' : 'plain' }}</em></b></div>
        <div class="kv"><span>breakup floor</span><b>{{ cfg.review.split_sim }}</b></div>
        <div class="kv"><span>face min score</span><b>{{ cfg.gates.face_min_score }}</b></div>
        <div class="kv"><span>body min height</span><b>{{ cfg.gates.body_min_rel_h }}</b></div>
        <div class="kv"><span>personness min</span><b>{{ cfg.gates.body_min_personness }}</b></div>
      </div>
    </section>

    <section class="card">
      <h2>Scan a folder</h2>
      <div class="row">
        <input class="input" v-model="scanPath" placeholder="/data/photos">
        <button class="btn primary" type="button" :disabled="scanBusy" @click="doScan">
          {{ scanBusy ? 'Scanning…' : 'Scan' }}
        </button>
      </div>
    </section>

    <section class="card">
      <h2>Upload photos</h2>
      <input ref="fileInput" type="file" accept="image/*" multiple
             :disabled="uploadBusy" @change="doUpload">
      <div class="muted" v-if="uploadBusy">Uploading…</div>
    </section>

    <section class="card">
      <h2>Re-cluster</h2>
      <button class="btn ghost" type="button" :disabled="reclusterBusy"
              @click="doRecluster($event.target)">
        {{ reclusterBusy ? 'Working…' : 'Re-cluster now' }}
      </button>
    </section>

    <section class="card">
      <h2>Reset options</h2>
      <div class="reset-list">
        <div class="reset-row" v-for="r in RESETS" :key="r[0]">
          <span class="reset-label">{{ r[0] }}</span>
          <button class="btn danger" type="button" @click="doReset(r[0], $event.target)">
            {{ r[0] === 'full' ? 'Full re-cluster…' : 'Reset ' + r[0] }}
          </button>
          <span class="reset-hint muted">{{ r[1] }}</span>
        </div>
      </div>
      <div class="muted" v-if="statusMsg">{{ statusMsg }}</div>
    </section>

    <section class="card">
      <h2>Labels</h2>
      <a class="btn ghost" :href="labelsHref">Export labels (training set)</a>
    </section>
  </div>
  `,
};
