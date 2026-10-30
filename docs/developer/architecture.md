# Architecture

---

## Processes

```
┌──────────────────────────┐   IPC (invoke)   ┌───────────────────────────┐
│ Renderer (Svelte 5)      │ ───────────────▶ │ Electron main             │
│ src/renderer             │                  │ src/main/index.ts         │
└──────────────────────────┘                  └─────────────┬─────────────┘
                                                            │ HTTP :5000 (fetchWithAuth)
                                              ┌─────────────▼─────────────┐
                                              │ Host backend (Flask)      │
                                              │ backend/app.py            │
                                              │  ParameterGraph, jobs,    │
                                              │  host operations          │
                                              └─────────────┬─────────────┘
                                                            │ EngineProvider
                                   ┌────────────────────────┴───────────────────────┐
                                   ▼                                                ▼
                     LocalEngine (in-process)                  RemoteEngine ──HTTP──▶ engine_service.py :5001
                     engine/local_engine.py                    engine/remote_engine.py  (Docker GPU host,
                                                                                          wraps a LocalEngine)
```

- **The renderer never calls the backend directly.** Each backend call goes through an
  `ipcMain.handle` in [src/main/index.ts](../../src/main/index.ts). Those handlers add the Cloudflare
  Access headers. The IPC surface is listed in [src/preload/index.ts](../../src/preload/index.ts).
- **Electron starts the backend.** It runs `backend/.venv/Scripts/python.exe -u backend/app.py` unless
  `RUN_LOCAL_BACKEND=false`. It then waits on `GET /health`. The renderer polls `/health` every 3 s.
  If `server_instance_id` changes, the backend has restarted, and the renderer unloads the project.
- **Media loads by path, not by HTTP.** The renderer asks `/audio_path/<id>` or `/image_path/<id>` for
  a path, then the main process reads that file from disk. This only works when the host backend
  shares the same filesystem.
- **[app.py](../../backend/app.py) is the orchestrator.** It holds the in-memory `ParameterGraph`,
  guarded by `graph_lock`. It also runs the job registries: `active_jobs` for engine jobs and
  `local_jobs` (the TaskManager) for host jobs.
- **The engine runs generative models.** `EngineProvider` holds a `LocalEngine` and any
  `RemoteEngine`s, and one of them is active. The first is created when a project is first loaded or
  created: remote if `ENGINE_URL` is set, local otherwise. `POST /engine` switches between them at
  runtime. Engines are kept after a switch and reused across project loads, with their data root
  pointed at the current project.

---

## The parameter graph

The graph lives in [backend/param_graph/](../../backend/param_graph/). It is a `networkx.DiGraph`
wrapped by `ParameterGraph(root)`, where `root` is the project directory.

### Project layout on disk

```
MyProject/
├── graph.json               # {"project_name", "graph": <cytoscape json>, "origin"?}
├── graph.json.bak           # rolling backup; load() falls back to it
├── embeddings.safetensors   # float32 vectors keyed "<node_id>|<clap|clip>"
├── generate/                # engine outputs, human-readable names
├── process/                 # host DSP outputs
├── cache/xx/<uid>           # content-addressed copies
└── .trash/<batch>/          # unreferenced files moved out by cleanup, with a manifest.json
```

**Removing an element doesn't delete its files.** Cleanup
([param_graph/cleanup.py](../../backend/param_graph/cleanup.py)) finds files in `generate/`,
`process/` and `cache/` that nothing references. It treats every string in the in-memory graph,
`graph.json` and `graph.json.bak` as a possible reference. It matches paths from the managed
directory onward, so a moved project keeps its files. It also keeps any file whose name, with its
suffixes stripped, equals a node ID or UID. Cleanup moves files to `.trash/` first, and refuses to
run while jobs are unfinished. Restoring a batch never overwrites a file that has reappeared.
Pre-migration backups and sync-conflict copies of `graph.json` are not protected.

**Saves are atomic.** `save()` works in this order:

1. Serialize the graph in memory.
2. Write a `graph.json.<hex>.tmp` file and `fsync` it.
3. Rotate the current `graph.json` to `.bak`.
4. `os.replace` the tmp file onto `graph.json`, retrying up to 5 times.

The retries exist because sync clients such as Proton Drive or OneDrive can hold the file open.

### Element types

Elements are `@dataclass(kw_only=True)` classes registered with `@register` in
[param_graph/registry.py](../../backend/param_graph/registry.py). The registry key is the element's
`type`, except for models, whose key is `model:<adapter>`.

| Kind | Types | Notes |
|---|---|---|
| Artifacts | `audio`, `image`, `latent`, `grating`, `individual`, `bundle` | Each has `name`, `context`, and a `file: Asset` |
| Collections | `group`, `directory`, `room` | Each has `member_ids`. Members point back to their collection through a `parent` attribute. |
| Models | `model:stable_audio_tools`, `model:stylegan2` | Each has `adapter`, `output_type`, `checkpoint: Asset`, `config` |
| Other | `local_path` | An external folder or file |

`context` records the operation and parameters that produced an element. Replicate, auto-labeling,
Repair Edges and exemplar inheritance all read it.

### Migrations

[utils/migrations.py](../../backend/utils/migrations.py) has no schema version. Each migration
detects old structure itself and is safe to run again. They all run on `/load_project`. To turn them
off, set `ENABLE_MIGRATIONS=false`. Migrations that rewrite content first save a
`graph.json.pre-<name>` backup.

---

## Storage and IDs

- **UIDs** are `<hex>.xxh3_64` content hashes, produced by `XXH3_64` in
  [utils/uid.py](../../backend/utils/uid.py). Hash inputs vary: tensors, state dicts, JSON, or whole
  directories.
- **`path_from_uid(uid)`** returns `cache/<first two hex chars>/<uid>`. Everything that crosses the
  host/engine boundary uses this layout.
- **Each result is stored twice.** Once a result is in CAS, `save_artifact_asset` copies it to
  `generate/` or `process/` under a readable name, and the graph node points at that copy.
- **Models stay where they are.** Model checkpoints are not copied into the project. Model nodes keep
  absolute paths.
- **Shared models** live in the engine's data root, apart from the cache:

  ```
  <data>/
  ├── cache/xx/<uid>              # anonymous, size-capped, evictable
  └── models/
      ├── xx/<uid>                # shared model files, never evicted
      └── shared_models.json      # registry: named references to those files
  ```

  Each registry entry gives `name`, `adapter` and `model_type`, plus `checkpoint`, optional
  `encoder` and optional `config_file` as file references. The config can be inline instead (as
  `config`). A file reference names a file in the store by `uid`, or gives a `path` to a file that
  already exists on the engine host. The format is shown in
  [shared_models.json.example](../../backend/shared_models.json.example), and the code is in
  [model_store.py](../../backend/engine/model_store.py).
- **Promotion and demotion.** **Share with Engine** calls `Engine.add_shared_model`. That moves the
  model's files from the cache into `models/` (a remote engine uploads them first if needed) and adds
  the registry entry. `remove_shared_model` does the reverse: the files go back into the cache and
  age out like any other entry. Either way the engine reloads its shared models without a restart.
- **Older setups.** A `backend/shared_models.json` (or `SHARED_MODELS_CONFIG`) from before the
  model store is converted by a startup migration and renamed to `*.migrated`.

> Don't change the storage layout or the CAS conventions without also updating `engine_service.py`
> and `/download_asset`. See AGENTS.md §5.

---

## Operations and jobs

`POST /execute_operation` is the single entry point. It is routed by
[operations/dispatcher.py](../../backend/operations/dispatcher.py):

| Kind | Examples | How it runs |
|---|---|---|
| Host, **immediate** | gain, normalize, slice, wrap_individual, express | Runs synchronously on the host. Results are written to CAS, then to `process/`, then added to the graph. |
| Host, **queued** | scatter, recombine, create_grating | Has a dedicated dispatcher, or runs through `TaskManager` (a thread pool of 8). Returns 202 and a job ID. |
| **Engine** (generative) | generate, invert | Resolves the model, inherits exemplar context, resolves gratings and individuals, stores `active_jobs[job_id]`, then calls `engine.execute`. Returns 202. |

**Job lifecycle.** The renderer polls `GET /job_status/<id>` every 1.5 s. When an engine job
completes, the poll handler does the following:

1. Ingests the result artifacts into `generate/`.
2. Merges the validated parameters into each artifact's `context`.
3. Handles phenotype duplicates.
4. Links inputs and adds group membership.
5. Saves the graph and refreshes embeddings.

> **Results only reach the graph when a client polls.** `active_jobs` exists only in memory. If the
> backend restarts mid-job, the job's result is lost.

---

## Embeddings and spring edges

After changes, the host computes embeddings: CLAP (`laion/clap-htsat-unfused`) for audio and CLIP
(`openai/clip-vit-base-patch32`) for images. It then rebuilds the `spring` edges using cosine kNN,
where k grows with log10(n).

If `backend/data/semantic_transitions_bank.pt` exists, each audio spring edge also gets a
`source_label`. The label is the nearest text label to the embedding difference between the two
clips. To build that file, run
[scripts/precompute_label_bank.py](../../backend/scripts/precompute_label_bank.py) on
`backend/data/semantic_transitions.txt`.

---

## Evolution internals

[evolution/lora_genome.py](../../backend/evolution/lora_genome.py) defines the genome. The
evolutionary operators come from the sibling `neutral-selection` library.

- **`LoRAGene`** holds `address`, `lora_down`, `lora_up`, `activation` (±1, stored as a tensor so
  crossover can inherit it) and `topology` (the kernel type, alpha and so on).
- **Genes are stored normalized.** `down` is scaled by `1/sqrt(fan_in)` and `up` by `sqrt(3/rank)`.
  As a result, one σ means the same relative perturbation in wide and narrow layers.
- **`LoRAMutation`** adds Gaussian noise to a gene with probability `mutation_rate`. It flips
  activations with probability `active_flips / len(genome)`. With `freeze_inactive` on, genes that are
  switched off are left unchanged.
- **Crossover** strategies map onto neutral-selection operators at different granularities: *atom*
  (any scalar), or *gene* (a whole layer). Blend crossover is a custom `FunctionCrossover`.
- **Genome files** are safetensors, format `"2"`. The hash of the state dict is the individual's ID,
  so identical genomes become the same node.
- **`express_to_grating`** turns a genome into a diffracture Grating. When generating, the
  `LocalEngine` attaches gratings, then individuals, as `diffracture.Actant` hooks, and detaches them
  in a `finally` block.

Lineage is recorded as `parent` edges, `scatter_operation.parent_id`, and `lineage.parent_ids`.
[operations/evolution/resolution.py](../../backend/operations/evolution/resolution.py) walks that
lineage to find exemplars and to collect exemplar presets.

---

## HTTP API (host, port 5000)

| Area | Endpoints |
|---|---|
| Project | `GET /health`, `POST /load_project`, `POST /create_project`, `GET /project`, `POST /fork_project` |
| Cleanup | `GET /cleanup`, `POST /cleanup/quarantine`, `POST /cleanup/restore`, `POST /cleanup/purge` |
| Rooms | `POST /rooms`, `POST /rooms/<id>/dissolve`, `POST /rooms/view` |
| Graph | `GET /graph`, `POST /graph/create_group`, `PUT\|POST /graph/update_group[/<id>]`, `POST /update_element`, `POST /save_node_positions`, `DELETE /element/<id>`, `DELETE /elements`, `POST /update_labels`, `POST /repair_edges`, `POST /update_embeddings` |
| Models | `GET /shared_models`, `DELETE /shared_models/<id>`, `POST /register_model`, `POST /export_shared_model`, `GET /model/<id>/layers`, `GET /adapter_config/<adapter>` |
| Gratings | `POST /register_grating`, `POST /create_grating` |
| Evolution | `POST /wrap_individual`, `POST /scatter_evolution`, `POST /recombine_evolution`, `POST /express_individual`, `POST /exemplar_presets` |
| Operations | `GET /operations`, `GET /sync_operations`, `POST /execute_operation` |
| Jobs | `GET /job_status/<id>`, `POST /jobs/<id>/cancel` |
| Sources | `POST /add_external_source`, `POST /rescan_source`, `POST /expand_path` |
| Media | `GET /audio_path/<id>`, `GET /audio/<id>`, `GET /audio_data/<id>`, `GET /image_path/<id>`, `GET /image/<id>`, `POST /export` |

The remote engine's API is listed in [remote-engine.md](remote-engine.md).

---

## Frontend

- Svelte 5 with runes only. The rules are in [AGENTS.md](../../.agents/AGENTS.md).
- [App.svelte](../../src/renderer/src/App.svelte) owns the layout and the panel routing.
- [ParameterGraph.svelte](../../src/renderer/src/components/graph/ParameterGraph.svelte) wraps
  cytoscape. It uses the fcose layout, the cxtmenu radial menus and popper. The radial menu entries
  are defined there, and the pinned operations are in `PINNED_OPERATIONS`.
- Node and edge styles are in [Style.ts](../../src/renderer/src/components/graph/Style.ts). The color
  tokens are in [base.css](../../src/renderer/src/assets/base.css).
- [DynamicForm.svelte](../../src/renderer/src/components/DynamicForm.svelte) renders the backend form
  configs (see [extending.md](extending.md#form-config-schema)).
- Jobs: [utils/job-management.ts](../../src/renderer/src/utils/job-management.ts).
  Rooms: [utils/rooms.ts](../../src/renderer/src/utils/rooms.ts).
