# Known issues

These issues were found while writing the documentation (October 2026). They have not been fixed yet.
Delete each entry once it is resolved.

---

## Bugs

| Issue | Where | Effect |
|---|---|---|
| `final_artifacts` / `is_batch` are referenced but never defined. They were lost in commit 85fdbc3. | [operations/dispatcher.py:192](../../backend/operations/dispatcher.py#L192) | **Every immediate DSP operation fails** (gain, normalize, slice, onset slice) with a `NameError` that surfaces as a 500. |
| The `/graph_tsne` route doesn't exist | [src/main/index.ts:831](../../src/main/index.ts#L831) | The toolbar's **Cluster** view returns a 404. `ParameterGraph.to_json(mode='cluster')` exists but has no route. |
| Success messages are shown in the error panel | `App.svelte` → `ErrorView` | "Export Successful" and similar messages appear in error colors. |
| `logMessage` is exposed in the preload, but nothing handles it in `ipcMain` | src/preload/index.ts | Calling it rejects. |
| Cancelling a *running* LocalEngine job only sets its status | engine/local_engine.py | The worker later overwrites the status with completed or failed. |
| `update_embedding` returns the undefined name `audio_artifact` on its error path | engine/local_engine.py:549 | The original error is replaced by a `NameError`. |
| The test venv can't import `neutral_selection` | backend/.venv | The editable installs point at the paths from before the rename (`Diffracture`, `NeutralSelection`). Re-run `npm.cmd run backend:setup`. |

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
  backend, and the production Python path is Windows-only.
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
