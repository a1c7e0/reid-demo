# Ente Face-Recognition Pipeline — Porting Specification

Source of truth: the Ente monorepo clone at `~/workspace/vendor/ente` (paths below are
relative to that root). The porting target is the **web client** implementation
(`web/packages/new/photos/services/ml/`), which contains the authoritative clustering and
person-management logic. The **mobile client** (`mobile/apps/photos/lib/…`) is cited where it
defines model assets, quality constants, or wire-format details shared with web. Every claim
below carries a `file:line` citation.

---

## 1. End-to-end face data processing flow

### 1.1 Overview

```
file (image or video thumbnail)
  → fetch source bytes                    [web/.../ml/blob.ts:22-60]
  → native inference (ONNX Runtime in an Electron "ML utility process")
        detection (YOLOv5s-face 640×640) → 5 landmarks + box + score
        alignment (similarity transform from landmarks)
        embedding (MobileFaceNet → 192-d, L2-normalized)
        quality (blurValue = Laplacian-based focus measure)
  → FaceIndex record (per file)           [web/.../ml/worker.ts:555-587]
  → remote storage: encrypted gzipped "mldata" file-data  [web/.../ml/ml-data.ts]
  → local storage: IndexedDB "ml" (face-index, clip-index, file-status)  [web/.../ml/db.ts]
  → face crops → blobCache("face-crops")  [web/.../ml/crop.ts]
  → clustering → FaceClusters (local) + cgroups (remote user entities)  [web/.../ml/cluster.ts, people.ts]
```

### 1.2 Indexing orchestration (web worker)

- All indexing runs in a web worker (`MLWorker`, `web/.../ml/worker.ts:70-262`), which talks to
  a native **Electron ML utility process** over a Comlink `MessagePort` (`worker.ts:85-101`,
  created via `createUtilityProcess(electron, "ml")` at `web/.../ml/index.ts:67`). Inference is
  entirely off-browser; **ML is desktop-only**: `isMLSupported = isDesktop`
  (`web/.../ml/index.ts:99`).
- Worker states: `"init" | "idle" | "tick" | "indexing" | "fetching"` (`worker.ts:59`). Idle
  backoff starts at 5 s and doubles to a max of 16 min: `idleDurationStart = 5`,
  `idleDurationMax = 16 * 60` (`worker.ts:61-62`), doubling at `worker.ts:185-186`.
- Backfill queue: up to 200 files per tick (`syncWithLocalFilesAndGetFilesToIndex(200)`,
  `worker.ts:222`), drawn from IndexedDB `file-status` rows in **descending fileID order**
  (higher IDs ≈ newer files; `readNextIndexableFileIDs`, `web/.../ml/db.ts:185-199`).
- Live uploads are indexed immediately (`onUpload`, `worker.ts:125-134`; only images:
  `index.ts:200-209`); clustering after unawaited indexing is debounced 30 s
  (`pDebounce(updateClustersAndPeople, 30 * 1e3)`, `web/.../ml/index.ts:187-190`).
- Fetch/decrypt of source files is parallelized across 4 tasks (`worker.ts:281`), but native
  inference is serialized through a single promise queue
  (`enqueueAnalyzeImage`, `worker.ts:538-552`).
- Files larger than `maxIndexableFileSize = 100 * 1000 * 1000` bytes are refused (videos are
  exempt because they use thumbnails) (`worker.ts:342`, `worker.ts:381-389`).
- Videos are indexed from their **thumbnail bytes**; images use original file bytes (or the
  in-flight upload item) (`blob.ts:28-30`, `blob.ts:22-60`). Native code performs all decoding;
  on native decode failure for formats needing JPEG conversion there is one JPEG-conversion
  retry (`analyzeImageWithConversionFallback`, `worker.ts:489-529`).
- Failure semantics: network errors retry; image-inherent errors are permanent and marked via
  `markIndexingFailed` (`worker.ts:395-398`, `worker.ts:414-417`, `db.ts:201-208`); a 409 on
  ML-data upload means "already exists" and is not a failure (`worker.ts:465-472`).
- Pet indexing is disabled: `petsIndexingEnabled = false` (`worker.ts:487`).

### 1.3 Native inference contract (detection → alignment → embedding → quality)

The native side is invoked as `electron.analyzeImage(request)`; the full IPC shape is
`web/packages/base/types/ipc.ts:139-175`:

- Request (`ipc.ts:152-158`): `{ fileID, bytes, runFaces, runClip, runPets, generateFaceCrops }`.
- Per-face result (`ipc.ts:160-166`):
  `{ faceId, detection: { score, boxXyxy: number[], keypoints: number[][] }, blurValue, embedding: Float32Array }`.
- Result (`ipc.ts:168-175`): `fileId`, `decodedImageSize {width,height}`, `faces?`,
  `faceCrops?: (Uint8Array|null)[]` (index-aligned with `faces`), `clip?: {embedding}`,
  `usedCoreml`, `usedWebgpu` (recorded into wire `flags`, see §1.6).

Pipeline stages (inferred from the record schema; the Rust implementation lives in the Ente
desktop native code, not in this clone):

1. **Detection** — YOLOv5s-face, 640×640 static-batch-1 ONNX model
   (`yolov5s_face_640_640_static_b1.onnx`, `mobile/.../ml_model_assets.dart:4-6`). Produces a
   score, an `xyxy` box and **5 keypoints** (left eye, right eye, nose, left mouth, right mouth —
   landmark indices 0..4 as used by `faceDirection`, `web/.../ml/face.ts:60-65`).
2. **Alignment** — a similarity transform derived from the 5 landmarks; the mobile wire record
   keeps `AlignmentResult { affineMatrix: 3×3, center: [x,y], size, rotation }`
   (`mobile/.../face_ml/face_alignment/alignment_result.dart:2-14`). The aligned face feeds the
   embedding model; a separate display crop is derived from the box (§1.5).
3. **Embedding** — MobileFaceNet (see §5), 192-d, **L2-normalized** ("Embeddings are
   normalized, so this is cosine similarity", `web/.../ml/cluster.ts:236`; "ML callers use
   normalized embeddings", `web/.../ml/math.ts:4-5`).
4. **Quality estimation** — one scalar `blurValue` per face (Laplacian-based focus measure;
   mobile thresholds are named `kLaplacian*`, `mobile/.../face_filtering/face_filtering_constants.dart:1-5`)
   plus the detection `score`. See §2.4 for exactly how quality gates clustering.

The web worker converts native typed arrays into the persisted `FaceIndex`
(`faceIndexFromAnalysisResult`, `worker.ts:555-587`): box `xyxy` → `{x,y,width,height}`
(`worker.ts:569-576`), keypoints → `landmarks: Point[]` (`worker.ts:577-580`),
`score: face.detection.score` (`worker.ts:582`), `blur: face.blurValue` (`worker.ts:583`),
`embedding: Array.from(face.embedding)` (`worker.ts:584`).

### 1.4 Face record schema and faceID

`Face` (`web/.../ml/face.ts:22-30`):

```ts
{
  faceID: string;                       // `${fileID}_${xMin}_${yMin}_${xMax}_${yMax}`
  detection: { box: Box; landmarks: Point[] };  // normalized [0,1] coords (face.ts:25)
  score: number;                        // detector confidence
  blur: number;                         // blurValue (higher = sharper)
  embedding: number[];                  // 192-d, L2-normalized
}
```

- `FaceIndex = { width, height, faces: Face[] }` (`face.ts:5-9`); remote variant adds
  `version, client, flags` (`face.ts:11-18`); local variant adds `fileID` (`face.ts:20`).
- **faceID format** (stable across equivalent reindexing): `fileID` + `_` + the four box
  coordinates, each rendered as `toStringAsFixed(5)` with the leading `0.` stripped, joined by
  `_` — e.g. `1234_01234_05678_09012_03456`
  (`mobile/.../face_ml/face_detection/detection.dart:148-176`; comment
  "prefixed with `${fileID}_`" at `web/.../ml/face.ts:23`). Parsing the fileID back is
  `fileIDFromFaceID` (`face.ts:44-51`).
- Versioning: `faceIndexingVersion = 1` (`face.ts:3`); remote indexes with
  `version >= faceIndexingVersion` are reused as-is, older ones are re-indexed
  (`worker.ts:350-360`). CLIP: `clipIndexingVersion = 1` (`web/.../ml/clip.ts:6`).

### 1.5 Face crops (display thumbnails)

- Best-effort, local-only ("Face crops are local-only and must be regenerated on other
  clients", `web/.../ml/crop.ts:7`); stored in `blobCache("face-crops")` keyed by faceID as
  JPEG blobs (`crop.ts:23-37`, cache reads at `index.ts:382-392`).
- Regenerated on demand when missing (`regenerateFaceCropsIfNeeded`, `index.ts:395-405`).
- Crop geometry (`extractFaceCrop`, `crop.ts:54-90`): denormalize the box, pad by
  `regularPadding = 0.4` of the face size on each side, shrinking symmetrically down to
  `minimumPadding = 0.1` when the padded crop would leave the image (`crop.ts:63-76`), clamp to
  image bounds (`crop.ts:78-81`), draw via `OffscreenCanvas` with high-quality smoothing and
  encode `image/jpeg` at `quality: 0.8` (`crop.ts:83-89`).

### 1.6 Storage

**Remote (cross-client)** — per-file `mldata` file-data entry (`worker.ts:452-473`):
a JSON object `{ face: RemoteFaceIndex, clip: RemoteCLIPIndex, ... }`, gzipped and encrypted
with the **file's own key** (`putMLData`, `web/.../ml/ml-data.ts:116-126`; fetch+decrypt+gunzip
at `ml-data.ts:66-102`; 4 MiB decompression cap `maxMLDataBytes = 4 * 1024 * 1024`,
`ml-data.ts:11`). Unknown top-level keys are preserved on upsert (`worker.ts:452-458`). Zod
schema for the remote face index: `ml-data.ts:27-50`. The `flags` bitmask records the runtime
used (`mlIndexFlagRuntimeRust = 1<<0`, `CoreML = 1<<1`, `WebGPU = 1<<2`;
`web/.../ml/index-flags.ts:2-14`, mirrored in `mobile/.../models/ml/ml_versions.dart:10-12`).

**Local (IndexedDB, DB name `"ml"` v1)** — object stores (`web/.../ml/db.ts:7-17`, created at
`db.ts:29-40`):
- `file-status` (`{fileID, status: "indexable"|"indexed"|"failed", failureCount}`)
- `face-index` (LocalFaceIndex), `clip-index` (LocalCLIPIndex)
- `face-cluster` (FaceCluster), `cluster-group`
Status and both indexes commit in one transaction ("Status must not say indexed unless both
indexes commit", `db.ts:88-101`). File removal keeps indexed rows for trashed files so a
restore does not re-index (`db.ts:144-153`).

**Local KV** — `rejectedClusters`: map `cgroupID → clusterID[]`
(`web/.../ml/kvdb.ts:4-27`; kept in KV "to avoid an IndexedDB migration", `kvdb.ts:8`).

**Remote clusters/people** — `cgroup` user entities (encrypted gzipped JSON,
`web/.../services/user-entity/index.ts:106-107`), synced via
`pullUserEntities`/`updateOrCreateUserEntities` (`user-entity/index.ts:77-144`). CGroup wire
schema `RemoteCGroupData` at `user-entity/index.ts:49-65`.

**Sync order** (must be preserved): "files, faces, cgroups, clusters, then people"
(`web/.../ml/index.ts:167`, `mlSync` at `index.ts:161-184`); clustering runs only after face
indexing completes and cgroups are pulled ("Face indexing must be complete and cgroups current
before this runs", `worker.ts:240`).

---

## 2. Clustering algorithm (greedy single-linkage) — exact specification

All of this is `web/packages/new/photos/services/ml/cluster.ts`. Entry point
`_clusterFaces(faceIndexes, localFiles, onProgress, reason)` (`cluster.ts:52-138`), invoked from
`MLWorker.clusterFaces` (`worker.ts:239-249`) which then calls `reconcileClusters`
(`cluster.ts:282-333`).

### 2.1 Face enumeration and filtering

`enumerateFaces` (`cluster.ts:140-153`) walks every `FaceIndex` and yields a `ClusterFace`
(`cluster.ts:47-50`: `Face` with `embedding: Float32Array` and `isBadFace: boolean`) **only if**
`face.blur > 10 && face.score > 0.8` (`cluster.ts:143`). Embeddings are converted to
`Float32Array` once because "dot products are the hot path" (`cluster.ts:146-147`).

### 2.2 Processing order

Faces are sorted **newest first** by the file's creation/photo sort time
(`sortFacesNewestOnesFirst`, `cluster.ts:155-175`; sort key
`fileCreationPhotoSortTime(file)` = `fileCreationPhotoDate(file).getTime()`,
`web/packages/media/file-metadata.ts:141-142`). Comment: "Newest faces first improves the
clustering heuristic" (`cluster.ts:62`). Missing local files fall back to sort time 0
(`cluster.ts:163-165`).

### 2.3 Seed state: existing assignments and rejection memory

Before the greedy pass (`cluster.ts:65-115`):

1. Remote `cgroup` entities are loaded and sorted by `updatedAt` **descending** so that "a face
   can occur in multiple remote clusters; newest assignment wins" (`cluster.ts:69-70`).
2. Each cgroup's `assigned: FaceCluster[]` is concatenated into the working `clusters` list.
   If the cgroup has `rejectedFaceIDs`, those faces are filtered out of the copied clusters and
   a **per-(face × cluster) rejection memory** is recorded:
   `rejectedClusterIDsForFaceID: Map<faceID, Set<clusterID>>` (`cluster.ts:72-90`) — for every
   rejected face, the IDs of all clusters assigned to that cgroup are remembered.
3. Local-only clusters (`savedFaceClusters()`) are appended after remote ones
   (`cluster.ts:92`), so remote assignments "precede and override duplicate local assignments"
   in `faceIDToClusterID` / `faceIDToClusterIndex` (`cluster.ts:98-106`).
4. `modifiedClusterIDs: Set<string>` tracks clusters touched in this run (`cluster.ts:108`).

### 2.4 Batching

The sorted face list is processed in overlapping windows: `batchSize = 10000` with
`offsetIncrement = 7500` ("Overlap batches so clusters can link across their boundaries",
`cluster.ts:117-130`).

### 2.5 The greedy single-linkage pass

`clusterBatchLinear` (`cluster.ts:194-280`):

1. The batch is split into already-clustered and unclustered faces; if nothing is unclustered,
   return early ("Common incremental case; avoids the O(n²) loop", `cluster.ts:210-214`).
2. **Already-clustered faces are placed first** in the working array: `clusteredFaces.concat
   (unclusteredFaces)` because "matches only look backward" (`cluster.ts:216-217`).
3. Iterate `i` over faces. Every 100 faces, report progress and `await wait(0)` to yield the
   worker (`cluster.ts:220-224`). Skip faces already assigned (`cluster.ts:226`).
4. For each unclustered face `fi`, scan **backward** over all previous faces
   `j = i-1 … 0` (`cluster.ts:232`) and compute `csim = dotProduct(fi.embedding, fj.embedding)`
   — cosine similarity because both embeddings are L2-normalized (`cluster.ts:236-237`,
   `math.ts:6-12`). Keep the nearest neighbor `nnIndex` subject to all of:
   - `csim` strictly improves on the running best (`csim <= nnCosineSimilarity → continue`,
     `cluster.ts:238`);
   - **quality-adaptive threshold** (the crux):
     ```ts
     const threshold = fj.isBadFace ? 0.84 : 0.76;   // cluster.ts:240
     if (csim < threshold) continue;                  // cluster.ts:241
     ```
     The threshold depends on the **candidate neighbor `fj`** (the already-placed face), not on
     the new face `fi`.
   - rejection memory: if `fi.faceID` has rejected clusters and `fj`'s cluster is among them,
     skip ("Never return a face to a cluster the user rejected", `cluster.ts:243-252`).
5. If a neighbor was found, `fi` joins the neighbor's cluster (push faceID, mark cluster
   modified) (`cluster.ts:258-268`); otherwise a new cluster is created with a fresh id from
   `newNonSecureID("cluster_")` (`cluster.ts:185`, `cluster.ts:269-278`).

This is greedy single-linkage: a face joins the cluster of its best previously-placed neighbor,
which may chain clusters transitively across the run. There is no centroid comparison, no
re-clustering/merge phase, and no split pass inside `_clusterFaces`.

### 2.6 What "quality" means and how it is computed

Per-face `isBadFace` (`cluster.ts:177-183`):

```ts
const isBadFace = (face: Face) =>
    face.blur < 50 ||
    (face.blur < 200 && face.score < 0.85) ||
    isSidewaysFace(face);            // faceDirection(face.detection) != "straight"
```

- `blur` is the native `blurValue`, a Laplacian-based sharpness measure (mobile thresholds
  `kLaplacianHardThreshold = 10`, `kLaplacianSoftThreshold = 50`,
  `kLaplacianVerySoftThreshold = 200`, sentinel `kLapacianDefault = 10000.0`;
  `mobile/.../face_filtering_constants.dart:1-5`). Higher = sharper.
- `score` is the detector confidence.
- Sideways: `faceDirection` (`web/.../ml/face.ts:60-95`) classifies the 5 landmarks as
  `"left" | "right" | "straight"`: upright check
  `max(leftEye.y, rightEye.y) + 0.5*eyeDistanceY < nose.y && nose.y + 0.5*mouthDistanceY <
  min(leftMouth.y, rightMouth.y)` (`face.ts:71-73`); "nose sticking out" left/right of both eyes
  and both mouth corners (`face.ts:75-81`); or upright with the nose within
  `0.2 * eyeDistanceX` of the left/right eye (`face.ts:83-86`). Anything not `"straight"` is
  sideways (`cluster.ts:182-183`).

So the quality-adaptive threshold is: **good face neighbor → cosine ≥ 0.76 suffices; bad face
neighbor (blurry / low-score / sideways) → cosine ≥ 0.84 required.** This is the web twin of
the mobile thresholds `kRecommendedDistanceThreshold = 0.24` /
`kConservativeDistanceThreshold = 0.16` (cosine *distance*; 1−0.24 = 0.76, 1−0.16 = 0.84,
`mobile/.../face_clustering/face_clustering_service.dart:75-78`) and of the mobile predicate
`isBadFaceForClustering` (`face_filtering_constants.dart:15-24`:
`faceScore < 0.80 || blur < 50 || (blur < 200 && score < 0.85) || isSideways`). Note the web
version's enumeration gate (`blur > 10 && score > 0.8`, `cluster.ts:143`) already excludes the
lowest-quality faces before clustering.

### 2.7 Persistence of clustering results (reconcile)

`reconcileClusters(clusters, modifiedClusterIDs, masterKey)` (`cluster.ts:282-333`):

- Any cgroup whose `assigned` contains a modified cluster is rewritten with the **new** cluster
  contents and pushed via `updateOrCreateUserEntities("cgroup", …)` (`cluster.ts:291-320`);
  a missing cluster id is a hard error ("Missing assignments must fail; dropping them would
  overwrite remote data", `cluster.ts:298-306`).
- Clusters belonging to any remote cgroup are **not** kept locally;
  `saveFaceClusters` stores only non-remote clusters (`cluster.ts:322-330`) — i.e. promoting a
  cluster into a person removes it from the local pool.
- After remote updates, `pullUserEntities("cgroup", …)` re-syncs (`cluster.ts:332`).
- `saveFaceClusters` clears the whole `face-cluster` store and re-puts the list
  (`web/.../ml/db.ts:215-221`).

### 2.8 Person (cgroup) creation / merge / split rules

There are exactly two person types (`people.ts:36-49`): a **cgroup** (named person, remote,
user-created) and a **cluster** (anonymous local cluster shown as "unnamed person"). Rules:

- **Create (naming a cluster)** — `addCGroup(name, cluster)`: posts a cgroup entity with
  `assigned: [cluster]`, `isHidden: false`, `isPinned: false`, `hideFromMemories: false`
  (`web/.../ml/index.ts:407-421`), then re-syncs. UI entry: AddPersonDialog (§6).
- **Create (ignore)** — `ignoreCluster(cluster)`: posts a cgroup with `name: ""`,
  `isHidden: true`, `assigned: [cluster]` (`index.ts:554-567`); an empty name also implies
  hidden for older mobile clients (`people.ts:111-112`).
- **Merge (cluster into person)** — `addClusterToCGroup(cgroup, cluster)`: appends the cluster
  to `assigned` and **removes its faces from `rejectedFaceIDs`** so previously rejected faces
  come back (`index.ts:423-439`). A whole file (without detected face) can be merged via
  `addManualFileAssignmentsToPerson` into `manuallyAssigned: number[]`
  (`index.ts:475-534`; duplicates against cluster-derived fileIDs are reported as
  `alreadyAssignedFileIDs`, `index.ts:493-511`).
- **Split (reject a cluster/face out of a person)** — via the review flow (§3.3):
  `rejectSavedChoice` unassigns the cluster and pushes its faceIDs into
  `cgroup.data.rejectedFaceIDs` (remote), permanently barring those faces from that person
  (`people.ts:519-523`, `people.ts:493-496`); the per-face × cluster rejection memory (§2.3)
  then blocks re-adding during future clustering.
- **Split (unassign without reject)** — `reset` removes the cluster from `assigned` and clears
  any rejection for it (`people.ts:525-528`, `unassignIfNeeded` at `people.ts:468-486` restores
  the cluster to the local pool: "Assigned clusters came from remote; restore them locally
  first", `people.ts:483-484`).
- **Delete person** — `deleteCGroup` deletes the remote entity (`index.ts:450-453`); its
  clusters fall back to local clusters and are re-clustered on the next run (since only
  non-remote clusters persist locally, `cluster.ts:322-330`). The UI calls this "Reset" with a
  confirmation (`PeopleHeader.tsx:132-141`) and "Show person" for ignored people
  (`PeopleHeader.tsx:216`).
- Within a cgroup, faces hidden by `rejectedFaceIDs` are filtered everywhere (people state
  `people.ts:114-122`, clustering seeds `cluster.ts:74-89`, suggestions `people.ts:260-264`).

---

## 3. Person management rules

### 3.1 CGroup data model and Person projection

`CGroupUserEntityData` (`web/.../ml/people.ts:22-34`):

```ts
{
  name?: string;                 // empty/undefined = unnamed; write "" when clearing
  assigned: FaceCluster[];       // unordered sets on the wire
  rejectedFaceIDs: string[];
  manuallyAssigned: number[];    // fileIDs assigned without a detected face
  isHidden: boolean;
  avatarFaceID?: string;
  isPinned?: boolean;
  hideFromMemories?: boolean;    // used by mobile only (people.ts:32-33)
}
```

`reconstructPeopleState()` (`people.ts:66-228`) builds `{ people, visiblePeople,
personByFaceID }`:

- Only faces whose file is in normal (non-hidden, non-deleted) collections count — hidden and
  deleted files "deliberately leave holes" (`people.ts:70-74`, `people.ts:76-90`).
- Faces of a person are sorted **newest first** by file sort time with **score as tie-break**
  (`personFacesSortedNewestFirst`, `people.ts:92-100`).
- A person with no remaining faces disappears (`people.ts:126-127`).
- `fileIDs` = distinct files of its faces **plus** `manuallyAssigned` fileIDs that still exist
  (`people.ts:129-133`).
- Empty `name` ⇒ `isHidden = true` (legacy mobile encoding) (`people.ts:109-112`).
- Ordering: cgroup people first (then cluster people), each group sorted pinned-first, then by
  `fileIDs.length` descending (`people.ts:189-200`).
- **Visibility** (`people.ts:202-214`): hidden cgroups are excluded; a local cluster person is
  visible only with **≥ 10 faces** (`p.cluster.faces.length < 10 → false`, `people.ts:209`;
  cf. `kMinimumClusterSizeSearchResult = 10`,
  `mobile/.../face_filtering_constants.dart:26`).
- `personByFaceID` maps every face (of all people) to its person (`people.ts:216-225`).

### 3.2 Naming and display-face selection

- **Naming**: `renameCGroup(cgroup, name)` (`index.ts:441-448`); creating a name is the same
  cgroup update (§2.8). Named people feed search:
  `setSearchPeople(filterNamedPeople(state.visiblePeople))` (`index.ts:321`,
  `people.ts:234-243`, `web/.../services/search/index.ts:63-64`).
- **Display face** (`people.ts:135-150`): if `avatarFaceID` is set and its file exists, use it;
  otherwise use the **most recent face** (per §3.1 sort). Legacy guard: a `avatarFaceID`
  without an underscore is a legacy `avatarFileID` and is ignored
  (`resolvedAvatarFaceID`, `people.ts:230-232`).
- **Pin**: `pinCGroup`/`unpinCGroup` set `isPinned` (`index.ts:455-468`); pinned people sort
  first (`people.ts:192-197`).
- Person sort options for UI: `PeopleSortBy = "count-desc" | "count-asc" | "name-asc" |
  "name-desc"` with pinned-first and named-before-unnamed ranking
  (`web/apps/photos/src/utils/people-sort.ts:3-56`).

### 3.3 Confirm / reject / review flow and its persistence

**Suggestions engine** — `_suggestionsAndChoicesForPerson(person, currentUserID)`
(`people.ts:254-438`), run in the ML worker (`worker.ts:256-261`):

- Person clusters = `assigned` minus `rejectedFaceIDs` (`people.ts:260-264`); per-cgroup
  **rejected cluster IDs** come from the local KV (`savedRejectedClustersForCGroup`,
  `people.ts:266-268`, `kvdb.ts:15-18`).
- Candidate pool = all **local** clusters (i.e. not yet assigned to any person), classified as
  rejected / singleton (`faces.length < 2`) / multi-face (`people.ts:294-311`).
- Scoring: sample up to `randomSample(..., 50)` person embeddings and up to 50 cluster
  embeddings (`people.ts:288-289`, `people.ts:318-320`), take all pairwise cosine similarities,
  and use the **median** ("Median similarity dampens a few misleading face pairs",
  `people.ts:331-334`).
- Selection cascades (`people.ts:356-377`):
  1. `strict_non_singleton`: multi-face clusters with median sim **> 0.48**
     (`strictMedianSimilarityThreshold`, `people.ts:291`);
  2. if empty, `relaxed_non_singleton`: **> 0.46** (`relaxedMedianSimilarityThreshold`,
     `people.ts:292`);
  3. if empty, `strict_with_singletons`: singletons at **> 0.48**.
- Each candidate gets up to **4** `previewFaces` (`people.ts:399`); at most **80** suggestions
  are returned (`suggestedClusters.slice(0, 80)`, `people.ts:431`).
- **Choices** = current assigned clusters (first one flagged `fixed: true`, not removable) plus
  previously rejected clusters (`assigned: false`), sorted by size (`people.ts:413-429`).

**Review UI contract** (`PeopleHeader.tsx:475-815`): each item toggles yes/no, mapping to
`PersonSuggestionUpdates = Map<clusterID, "assign" | "rejectSuggestion" |
"rejectSavedChoice" | "reset">` (`people.ts:440-443`; reducer mapping at
`PeopleHeader.tsx:538-561`). **Persistence semantics** (`_applyPersonSuggestionUpdates`,
`people.ts:445-556`):

| Update | assigned | rejectedClusterIDs (local KV) | rejectedFaceIDs (remote) |
|---|---|---|---|
| `assign` | push cluster (+ un-reject) (`people.ts:509-512`) | remove if present (`people.ts:498-505`) | — |
| `rejectSuggestion` | unassign if present (`people.ts:514-517`) | add (`people.ts:488-491`) | — |
| `rejectSavedChoice` | unassign (`people.ts:519-523`) | add | append the cluster's faceIDs (`people.ts:493-496`) |
| `reset` | unassign (`people.ts:525-528`) | remove | — |

Assignments/rejections that change `assigned` or `rejectedFaceIDs` are written to the remote
cgroup immediately (`people.ts:532-547`); cluster-level rejections are saved locally in the KV
(`people.ts:549-551`). Un-assigning restores the cluster to the local store
(`people.ts:483-484`, `people.ts:546`).

### 3.4 Per-file people UI data

`getAnnotatedFacesForFile` (`index.ts:342-369`) returns face→person annotations for a file's
info panel, sorted named-persons-first then by person size (`index.ts:363-367`).
`faceCrop(faceID, file)` returns a cache-owned object URL, regenerating crops if needed
(`index.ts:371-405`; callers must not revoke, `index.ts:372`).

---

## 4. Constants and defaults (with citations)

### Clustering (`web/.../ml/cluster.ts`)

| Constant | Value | Meaning | Location |
|---|---|---|---|
| enumeration gate | `blur > 10 && score > 0.8` | face is clusterable at all | `cluster.ts:143` |
| good-neighbor threshold | `0.76` | min cosine when neighbor `!isBadFace` | `cluster.ts:240` |
| bad-neighbor threshold | `0.84` | min cosine when neighbor `isBadFace` | `cluster.ts:240` |
| isBadFace | `blur < 50 \|\| (blur < 200 && score < 0.85) \|\| sideways` | per-face quality | `cluster.ts:177-180` |
| batch size / stride | `10000` / `7500` | overlapping windows | `cluster.ts:119-120` |
| progress yield | every `100` faces, `wait(0)` | keep worker responsive | `cluster.ts:220-223` |
| cluster ID prefix | `"cluster_"` | `newNonSecureID` | `cluster.ts:185` |

### Face geometry (`web/.../ml/face.ts`)

| Constant | Value | Location |
|---|---|---|
| `faceIndexingVersion` | `1` | `face.ts:3` |
| upright margins | `0.5 * eyeDistanceY`, `0.5 * mouthDistanceY` | `face.ts:71-73` |
| nose-near-eye margin | `0.2 * eyeDistanceX` | `face.ts:84-86` |

### Quality thresholds (mobile, shared semantics)

| Constant | Value | Location |
|---|---|---|
| `kLaplacianHardThreshold` | `10` | `mobile/.../face_filtering_constants.dart:1` |
| `kLaplacianSoftThreshold` | `50` | `…:2` |
| `kLaplacianVerySoftThreshold` | `200` | `…:3` |
| `kLapacianDefault` (unknown blur) | `10000.0` | `…:5` |
| `kMinFaceDetectionScore` | `0.5` | `…:7` |
| `kMinimumFaceShowScore` | `0.75` | `…:9` |
| `kMinimumQualityFaceScore` | `0.80` | `…:11` |
| `kMediumQualityFaceScore` | `0.85` | `…:12` |
| `kHighQualityFaceScore` | `0.90` | `…:13` |
| `kMinimumClusterSizeSearchResult` | `10` | `…:26` |
| mobile clustering distance thresholds | `0.24` recommended / `0.16` conservative (Δ0.08) | `mobile/.../face_clustering_service.dart:75-78` |

### People / suggestions (`web/.../ml/people.ts`)

| Constant | Value | Location |
|---|---|---|
| visible cluster-person min size | `10` faces | `people.ts:209` |
| person-embedding sample | `50` | `people.ts:289` |
| per-cluster sample | `50` | `people.ts:318` |
| strict median similarity | `0.48` | `people.ts:291` |
| relaxed median similarity | `0.46` | `people.ts:292` |
| preview faces per cluster | `4` | `people.ts:399` |
| max suggestions | `80` | `people.ts:431` |
| singleton cluster | `< 2` faces | `people.ts:306` |
| score tie-break | `b.score - a.score` | `people.ts:99` |

### Worker / storage (`web/.../ml/worker.ts`, `db.ts`, `ml-data.ts`, `crop.ts`, `clip.ts`)

| Constant | Value | Location |
|---|---|---|
| `idleDurationStart` | `5` (s) | `worker.ts:61` |
| `idleDurationMax` | `16 * 60` (s) | `worker.ts:62` |
| backfill batch | `200` files | `worker.ts:222` |
| parallel fetch/decrypt tasks | `4` | `worker.ts:281` |
| `maxIndexableFileSize` | `100 * 1000 * 1000` bytes (non-video) | `worker.ts:342` |
| `petsIndexingEnabled` | `false` | `worker.ts:487` |
| cluster-recompute debounce | `30 * 1e3` ms | `index.ts:187-190` |
| status throttle | `2000` ms | `index.ts:297` |
| "phaseFailed" heuristic | `failedCount > indexedCount && failedCount > 500` | `index.ts:274` |
| ML DB name / version | `"ml"` / `1` | `db.ts:29` |
| trash retention of indexed rows | keep `"indexed"` rows for trashed files | `db.ts:146-151` |
| `maxMLDataBytes` (gunzip cap) | `4 * 1024 * 1024` | `ml-data.ts:11` |
| crop padding | `regularPadding = 0.4`, `minimumPadding = 0.1` | `crop.ts:64-65` |
| crop JPEG quality | `0.8` | `crop.ts:89` |
| CLIP match threshold | `0.175` (cosine) | `clip.ts:37` |
| index flags | `rust=1<<0`, `coreml=1<<1`, `webgpu=1<<2` | `index-flags.ts:2-4` |
| ML enable keys | localStorage `"mlEnabled"`, remote flag `"faceSearchEnabled"` | `index.ts:138-148` |

### Mobile version constants (`mobile/.../models/ml/ml_versions.dart`)

`faceMlVersion = clipMlVersion = clusterMlVersion = petMlVersion = 1`,
`minimumClusterSize = 2` (`ml_versions.dart:3-7`); `embeddingFetchLimit = 200`,
`fileDownloadMlLimit = 5` (iOS) / `10`, `maxFileDownloadSize = 100000000`
(`ml_versions.dart:14-16`).

---

## 5. Face embedding model

- **Detector**: YOLOv5s-face, input **640×640**, static batch-1 ONNX —
  `yolov5s_face_640_640_static_b1.onnx`, SHA-256
  `e047647409403d52696035ecd445792173e50d7fbdcccac97b958a585db9aa3d`
  (`mobile/.../ml_model_assets.dart:4-6`). Older variant `yolov5s_face_640_640_dynamic.onnx`
  is listed as a retired model (`mobile/.../remote_assets_service.dart:214-216`).
- **Embedding**: **MobileFaceNet** — `mobilefacenet_portable_static_b1.onnx`, SHA-256
  `0763fc33f54e138476194da95987e133b3e976075a6b1d3e1b2caedb251b1a36`
  (`mobile/.../ml_model_assets.dart:18-21`; retired predecessor `mobilefacenet_opset15.onnx`,
  `remote_assets_service.dart:215`). Output dimension **192**
  (`ClusterCentroidVectorDB.embeddingDimensions = 192`,
  `mobile/.../db/ml/cluster_centroid_vector_db.dart:11`; `Vector.filled(192, 0)`,
  `mobile/.../face_ml/feedback/cluster_feedback.dart:2174`). Embeddings are **L2-normalized**
  before storage (`web/.../cluster.ts:236`, `web/.../math.ts:4-5`).
- **CLIP** (adjacent, for search): MobileCLIP-S2 image/text, 512-d
  (`mobile/.../ml_model_assets.dart:33-67`, `mobile/.../db/ml/clip_vector_db.dart:7`).
- **Where it runs — on-device, not server-side.** Current Ente uses **ONNX Runtime**, not
  TFLite: mobile links a custom prebuilt ONNX Runtime 1.28.1 (`io.ente.onnxruntime:
  onnxruntime-webgpu-android:1.28.1-r1`, `mobile/apps/photos/android/app/build.gradle:188`;
  iOS static libs `mobile/apps/photos/ios/Podfile:36`; Rust `ort` crate,
  `mobile/apps/photos/rust_builder/ios/ente_photos_frb.podspec:28`), with optional CoreML /
  WebGPU acceleration recorded in `flags` (`index-flags.ts:2-4`,
  `ipc.ts:174-175`). The web/desktop app runs the same models in a **native Electron ML
  utility process** (`UtilityProcessType = "ml"`, `web/packages/base/types/ipc.ts:137`;
  `createUtilityProcess(electron, "ml")`, `web/.../ml/index.ts:67`) — pure web browsers cannot
  run ML at all (`isMLSupported = isDesktop`, `index.ts:99`). The server never computes face
  embeddings; it only stores encrypted `mldata` blobs (`web/.../ml/ml-data.ts:116-126`) and
  cgroup user entities.
- **Model distribution**: downloaded from `https://models.ente.com/` with SHA-256 verification
  (`kModelBucketEndpoint`, `mobile/.../ml_model.dart:10`; hashes above). A port should bundle
  equivalent ONNX models (YOLOv5s-face 640×640 + MobileFaceNet 192-d) or substitute compatible
  ones and re-tune thresholds.
- **Alignment input spec**: the embedding input is the landmark-aligned face crop produced by a
  similarity transform (`AlignmentResult {affineMatrix 3×3, center, size, rotation}`,
  `mobile/.../alignment_result.dart:2-14`). The exact embedding-network input resolution is
  defined in the native ONNX pipeline (not present in this clone); MobileFaceNet's standard
  input is 112×112×3 — treat that as an assumption to verify against the model file.

---

## 6. Web UI structure — people & review screens

All UI is React + MUI (i18n via `i18next`). Primary files:

### 6.1 Gallery integration

- `web/apps/photos/src/pages/gallery.tsx` — hosts the people bar mode; `onSelectPerson`
  dispatches `showPerson` (`gallery.tsx:1525-1527`); `AllPeople` receives `people`/`allPeople`
  from the people view (`gallery.tsx:1982-1989`); people appear in search only when ML is
  enabled and named people exist (`gallery.tsx:255-264`).
- `web/apps/photos/src/components/gallery/reducer.ts` — gallery state machine: view type
  `"people"` with `{ people, visiblePeople, activePerson }` (`reducer.ts:53-65`); actions
  `showPeople` / `showPerson` (`reducer.ts:807-840`); `derivePeopleView`
  (`reducer.ts:1518-1577`) filters temporarily deleted/hidden files out of person file counts
  and keeps a selected sub-10-face cluster visible via `extraVisiblePerson`
  (`reducer.ts:1543-1551`, "Keep a selected small cluster visible until the user leaves this
  view").
- `web/apps/photos/src/components/gallery/BarImpl.tsx` — the "People / Albums" toggle and a
  horizontally scrollable strip of person tiles (`BarImpl.tsx:317-320`, people item rendering
  at `BarImpl.tsx:393-460`), plus PeopleSortOptions (`BarImpl.tsx:189-219`).
- `web/apps/photos/src/components/PeopleSortOptions.tsx` + `web/apps/photos/src/utils/people-sort.ts`
  — sort menu (count/name, asc/desc; pinned first).
- `web/apps/photos/src/components/sidebar/preferences/MLSettings.tsx` — enable/disable ML.

### 6.2 People overview dialog — `AllPeople.tsx`

`web/apps/photos/src/components/Collections/AllPeople.tsx` (dialog, right-anchored, rounded
32px paper, slide-up transition; `AllPeople.tsx:195-242`):

- Header: title "people" + count, `PeopleSortOptions`, search field
  (`Title`, `AllPeople.tsx:285-333`).
- Content: virtualized (`react-window` `VariableSizeList` + `AutoSizer`) 3-column grid
  (`GridColumns = 3`, `GridGap = 8`, `GridPaddingInline = 20`, `AllPeople.tsx:265-272`); rows
  are `PeopleRow` items (`AllPeople.tsx:423-481`); a "show more/less faces" button row expands
  the low-count "extra people" section (`AllPeople.tsx:607-645`).
- `PersonCard` (`AllPeople.tsx:656-716`): `ItemCard` tile with `coverFaceID` face crop, name
  overlay (3-line clamp), photo count, pinned icon; hover-revealed `PersonActionMenu`
  (`AllPeople.tsx:740-846`): cgroup → Rename / Pin-Unpin; cluster → Add name / Ignore.
- `AddPersonDialog` (`AllPeople.tsx:876-1012`): pick an existing person or create new
  ("new person" dashed tile → `SingleInputDialog` for the name); choosing calls
  `addClusterToCGroup`, creating calls `addCGroup` (`AllPeople.tsx:902-919`).

### 6.3 Person header + review dialog — `PeopleHeader.tsx`

`web/apps/photos/src/components/gallery/PeopleHeader.tsx` (rendered above the person's photo
grid):

- `PeopleHeader` routes by person type (`PeopleHeader.tsx:89-112`):
  - `CGroupPersonHeader` (`:118-205`): summary (name + file count) and overflow menu →
    **"Review suggestions"** (opens `SuggestionsDialog`), **Rename** (`SingleInputDialog`),
    **Pin/Unpin**, **Reset** (mini-dialog confirm → `deleteCGroup`, `:132-141`).
  - `IgnoredPersonHeader` (`:211-235`): "ignored" label + **Show person** (`deleteCGroup`).
  - `ClusterPersonHeader` (`:242-302`): "unnamed person", Add-name affordances, and
    **Ignore** with confirm → `ignoreCluster` (`:254-263`).
- `AddPersonDialog` (`:309-424`): searchable list of existing cgroup people (large tiles with
  face-crop covers, `PersonButton` `:459-473`) + "new person" → name input; full-screen below
  490px (`:316`). Empty people list redirects straight to naming (`:350-356`).
- `SuggestionsDialog` (`:573-749`) — the review screen:
  - On open, fetches `suggestionsAndChoicesForPerson` (`:594-614`); reducer state machine
    (`:477-571`) with `activity: "fetching" | "saving"`, `choices`, `suggestions`, `updates`.
  - Header toggles between "review suggestions" and "saved choices" (restore icon) when there
    is more than one choice (`:674-691`).
  - Each row (`SuggestionOrChoiceList`, `:757-792`): photo count ("Face count stands in for
    photo count"), `SuggestionFaceList` (up to 4 circular face crops, 87px), and an exclusive
    yes/no `ToggleButtonGroup`; `fixed` items (the anchor choice) hide the toggles (`:775`).
  - Close with unsaved changes prompts discard (`:616-631`); Save calls
    `applyPersonSuggestionUpdates` (`:636-645`).

### 6.4 Shared components

- `web/packages/new/photos/components/PeopleList.tsx`:
  - `SearchPeopleList` (`:18-45`) — up to 6/7 circular 87px face avatars in the search dropdown.
  - `FilePeopleList` (`:84-140`) — 65px circular annotated faces + manually assigned people +
    "Add person" button (used in the file info panel).
  - `SuggestionFaceList` (`:203-218`) — 87px circular crops for review rows.
  - `FaceCropImageView` (`:242-270`) — async face-crop loader with circular skeleton
    placeholder; missing crops regenerate from the file.
- `web/packages/new/photos/components/AssignPersonDialog.tsx` (`:19-151`) — search-filtered
  grid of large person tiles (auto-fill 150px columns) used to assign a whole file to a person
  (from `web/packages/gallery/components/FileInfo.tsx:387-401`, which lists a file's detected
  faces and calls `addManualFileAssignmentsToPerson`, `FileInfo.tsx:227-231`).
- `web/packages/new/photos/components/Tiles.tsx` — `ItemCard` with `coverFaceID`: renders the
  face crop as tile cover via `faceCrop(coverFaceID, coverFile)` (`Tiles.tsx:16-57`); tile
  variants `LargeTileButton`, `LargeTileTextOverlay`, `LargeTileCreateNewButton`
  (`Tiles.tsx:119-164`), plus `CollectionTile*` primitives in
  `web/apps/photos/src/components/CollectionDialog/Primitives`.

### 6.5 Interaction flows to replicate

1. **People bar → person**: tap person tile → `showPerson` → gallery filters to `person.fileIDs`
   → `PeopleHeader` shows person actions.
2. **Name an unnamed cluster**: "Add name" → pick existing person (merge cluster) or create
   named person (cgroup with one cluster) → `mlSync("add-cgroup" | "add-cluster-to-cgroup")`.
3. **Review suggestions**: person overflow → Review suggestions → toggle yes/no per cluster →
   Save → remote cgroup update + local rejected-clusters KV + `mlSync`.
4. **Ignore**: cluster overflow → Ignore (confirm) → hidden cgroup with empty name; undo via
   IgnoredPersonHeader → Show person (deletes the cgroup).
5. **Reset person**: confirm → `deleteCGroup` → clusters return to the local pool and re-cluster
   on next sync.
6. **Manual assignment**: file info → Add → AssignPersonDialog → `manuallyAssigned` fileIDs.

---

## 7. Porting checklist (derived invariants)

1. Preserve faceID format and its fileID prefix (`detection.dart:148-176`, `face.ts:44-51`).
2. Persist embeddings L2-normalized; similarity = dot product (`cluster.ts:236-237`).
3. Clustering order is newest-first, matches look only backward, already-assigned faces first
   (`cluster.ts:62-63`, `cluster.ts:216-217`, `cluster.ts:232`).
4. Threshold is chosen by the **neighbor's** quality: 0.76 good / 0.84 bad (`cluster.ts:240`).
5. User rejections are remembered per (face × cluster) and never re-attached
   (`cluster.ts:72-90`, `cluster.ts:243-252`).
6. Re-clustering is full-rebuild + reconcile, batched 10000/7500 (`cluster.ts:117-130`,
   `cluster.ts:322-330`).
7. People state derives from cgroups (remote) + local clusters, with cluster-persons shown only
   at ≥10 faces (`people.ts:202-214`).
8. Rejections are two-tier: cluster-level (local KV, review history) and face-level (remote
   `rejectedFaceIDs`, permanent per person) (`people.ts:507-551`).
