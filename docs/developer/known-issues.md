# Known issues

These issues were found while writing the documentation (October 2026). They have not been fixed yet.
Delete each entry once it is resolved.

---

## Bugs

| Issue | Where | Effect |
|---|---|---|
| Stopping a *running* LocalEngine job doesn't interrupt it | engine/local_engine.py | The job stays cancelled and its result is discarded, but the operation still runs to the end, and queued jobs wait for it. Interrupting it needs a cancel check in the adapters' per-step hooks. |
| Loading a project while jobs are running moves their output | engine/engine_provider.py | Engines are reused across projects, and `set_data_root` repoints them immediately. A job still running writes its output under the newly loaded project, whose graph then receives the result. Unlike `POST /engine`, project loads don't check for unfinished jobs. |

---

## Inconsistencies

- **Form defaults and backend defaults differ.** UI users get the form values; API callers get the
  backend fallbacks.

  | Parameter | Form | Backend |
  |---|---|---|
  | scatter offspring_size | 10 | 5 |
  | scatter active_flips | 0.0 | 1.0 |
  | recombine mutation_rate | 0.01 | 0.1 |
  | generate_exemplars | true | false |

- **"Exemplar" has two meanings.** It is an individual's rendered output, and also Diffracture's
  "exemplar classifier" used for channel clustering.
- **"Breakout" has two meanings.** The UI's *Break Out* creates a room, but docstrings in `transfer.py`
  and `graph.py` use "breakout" to mean forking to a project.
- **Recombine's "Wholesale Layer Splicing (Depth 1)"** actually uses gene granularity with one cut.
  The *Multi-Tier* options cut at atom granularity, so a cut can fall inside a tensor.
- **The `wrap_individual` form** restricts its precursor to audio, but its `initiator_types` also
  include image, grating and latent.
- **No backend code creates `bundle` nodes,** although the frontend and recombine both handle them.
- **Auto-exemplars from several parents** only inherit the *first* parent's exemplar context.

---

## Dead or stale code

- `GraphControls.svelte`, `GraphLegend.svelte`, `ExecutionView.svelte`, `Versions.svelte` and the
  expand-collapse options in `graph/Options.ts` are not used anywhere.
- AGENTS.md §5 refers to `temp_{job_id}` paths, which no longer exist.
- `ENGINE_IDLE_TIMEOUT` defaults to 6000 s, but its comment says 30 minutes.
- `backend/tmp/cache/*` is tracked in git even though `tmp` is gitignored.
- `param_graph/registry.py` prints `DEBUG: Registering ...` for every element class.

---

## Packaging and environment

- **Packaged builds don't ship the backend.** `electron-builder.yml` has no `extraResources` for the
  backend.
- **macOS setup is untested.** `backend/setup.sh` and the `bin/python` launch path haven't been run
  on a Mac. The backend has no MPS device selection, so Macs fall back to the CPU.
- **`DATA_ROOT` is unused.** Electron sets it, but Python never reads it.
- **The dev backend window isn't closed on quit.** It is started with `cmd /c start`.
- **The fallback for `LOCAL_DATA_PATH` differs** between `app.py` (`~/.stemma2a_data`) and
  `engine_service.py` (it raises an error).
- **`.env.example` is misleading for compose.** It suggests `ENGINE_URL=http://127.0.0.1:5001`, but
  compose publishes no ports.
- **The semantic label bank is gitignored.** `backend/data/` is ignored, so a fresh clone gets no
  spring-edge labels, and nothing warns about it.
- **Fail-fast violations** (AGENTS.md §4):
  - `operations/__init__.py` swallows import errors.
  - `RemoteEngine.get_supported_operations` silently falls back to the local operations list.
