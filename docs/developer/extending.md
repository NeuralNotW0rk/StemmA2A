# Extending StemmA2A

---

## Adding a host operation

Host operations run on the host backend, not the engine. Examples are DSP effects and evolution
steps. They live in [backend/operations/](../../backend/operations/). Every submodule is imported
automatically at startup.

1. Create `backend/operations/<area>/<my_op>.py`:

   ```python
   from ..base import SyncOperation
   from ..registry import register

   @register
   class MyOperation(SyncOperation):
       @property
       def name(self) -> str:
           return "my_op"                 # unique; also the JSON filename

       @property
       def description(self) -> str:
           return "One line shown in the Operations palette."

       @property
       def initiator_types(self) -> list:
           return ["audio"]               # node types whose menus offer this op

       def execute(self, **kwargs):
           source = kwargs["source_audio_element"]   # node fields arrive as <name>_element
           ...
           return [(artifact, tensor)]    # list of (Artifact blueprint, data)
   ```

   [operations/audio/gain.py](../../backend/operations/audio/gain.py) is a complete minimal example.

2. Put `my_op.json` next to the Python file. It describes the form, as shown in the
   [form config schema](#form-config-schema) below.

3. Choose how the operation runs:
   - **`execution = "immediate"`** (the default): the operation runs synchronously. `execute()`
     returns `[(artifact, tensor)]`. The dispatcher saves each tensor to CAS, copies it into
     `process/`, adds it to the graph with `source` edges, and refreshes embeddings.
   - **`execution = "queued"`**: implement `execute_task(job_id, **payload)`. The operation then runs
     on the `TaskManager` thread pool, and the UI shows it as a job. The generic queued path passes
     only the raw payload. If your operation needs the graph, the lock or the engine, add a dedicated
     dispatcher, as `scatter` and `recombine` do.

4. Give each artifact a **content-hash ID**, from `XXH3_64().from_tensor(...)`. Record the parameters
   in `context`, for example `{"operation": ..., "params": ..., "source_id": ...}`. Replicate and the
   auto-labeling features read this context.

> `operations/__init__.py` currently swallows import errors. If your operation doesn't show up in the
> palette, import the module manually to see the error.

---

## Form config schema

A config file is either a **list of fields**, or an **object** with `form_config` and
`initiator_types`. The other keys (`name`, `category`, `execution`) are ignored; the Python properties
take precedence.

```json
{
  "initiator_types": ["individual", "group", "audio"],
  "form_config": [
    { "name": "parents", "type": "node-list", "label": "Seed Individual(s)",
      "filter": { "type": "individual" }, "required": true, "minItems": 1 },
    { "name": "offspring_size", "type": "integer", "label": "Scatter Count (N)",
      "defaultValue": 10, "min": 1, "max": 100 }
  ]
}
```

| Field type | Renders as |
|---|---|
| `text` / `string` | Text input. Unknown types also fall back to this. |
| `textarea` | Multi-line text |
| `integer` / `float` | Number input. Integers are rounded. |
| `boolean` | Checkbox |
| `select` | Dropdown, populated from `options` |
| `file` / `directory` | Path field with a Browse button. `filters` restricts the file types. |
| `node` | Graph node picker. `filter: {"type": ...}` restricts which nodes can be picked. |
| `node-list` | List of node pickers |

| Key | Meaning |
|---|---|
| `defaultValue`, `placeholder`, `description` | Shown in the form |
| `required`, `minItems` | Disable the Run button until satisfied |
| `min`, `max`, `step` | Numeric bounds |
| `randomizable` | Adds a 🎲 button. The field is also randomized when the form first loads. |
| `allowSequence` | Adds sequence mode, with the `a..b:step` syntax |
| `show_if` | Shows the field only when a condition holds. An array of conditions means OR. The value `"exists"` matches any value. |
| `conditionalDefaults`, `conditionalProps` | Change the default, label or placeholder depending on other fields |

On the backend, [utils/form.py](../../backend/utils/form.py) validates parameters by building a
Pydantic model from the same field list.

> `forms.ts` declares a `number` type, but the renderer has no branch for it, so a `number` field
> shows as a plain text input. Use `integer` or `float`.

---

## Adding a generative (engine) operation

Generative operations run on the engine, which may be remote. To add one:

1. Add a method decorated with `@operation(name=..., initiator_types=[...])` to the model adapter
   in [engine/model_adapters/](../../backend/engine/model_adapters/). With `is_standard=True`, the
   operation is merged across adapters under one name, the way `generate` and `invert` are.
2. Add a `"<name>"` section to the adapter's JSON, for example `stable_audio_tools.json`.
3. Add a `_<name>_logic` method to `LocalEngine`. Without it, the engine raises "not supported".
4. If the operation returns more than one artifact, return
   `{"artifacts": [...], "links": [{source, target, relation}]}`. The format is defined in
   [engine/results.py](../../backend/engine/results.py).

---

## Adding a model adapter

1. Subclass the base adapter in [base_adapter.py](../../backend/engine/model_adapters/base_adapter.py).
   Set `self.name` to the registry key. The adapter's JSON file must be named `<self.name>.json`.
2. Implement `register_model`, which must hash the checkpoint into a stable ID, plus your operations.
3. Register the adapter in `Engine.adapter_registry` in [engine/engine.py](../../backend/engine/engine.py).
4. Add a `Model` subclass under
   [param_graph/elements/models/](../../backend/param_graph/elements/models/), registered with the
   key `model:<adapter>`.
5. Add the adapter to the dropdown in `ImportModelView.svelte`.

---

## Tests

The backend tests are `unittest` files in [backend/tests/](../../backend/tests/).

```powershell
npm.cmd run test:backend
# single file, from backend/
.\.venv\Scripts\python.exe -m unittest tests/test_recombination.py
```

`npm.cmd run typecheck` covers the frontend.
