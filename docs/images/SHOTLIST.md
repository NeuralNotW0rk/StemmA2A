# Screenshot shot list

To add a screenshot, save it in this folder under the exact filename shown below. The docs already
link to these filenames, so the image appears as soon as the file exists.

**Before you start**

- Set the window to about 1400×900 so shots are consistent.
- Use one demo project with real content for every shot, so the shots look consistent.
- Crop panel shots to the panel plus a little of the canvas around it.

---

## Priority 1: the user guide depends on these

| File | Used in | Setup / what to capture |
|---|---|---|
| `overview.png` | concepts | A mid-sized project zoomed to fit. It should show a model, a grating bound to it, a group of generated clips, and a short evolved lineage (gen-0 individual → Scatter group → Recombination group). Turn on Edge Labels. |
| `main-window.png` | interface | The full window: one audio node selected (details panel open, player at the bottom) while one job is running (Jobs panel visible). |
| `evolution-lineage.png` | workflows | Close-up of an individual box holding exemplars, with `[fit: X]` labels. Include one child marked `[Duplicate]` if you have one. |
| `radial-menu-model.png` | interface | Right-click held on a model node, with the radial menu open. |
| `jobs-panel.png` | interface | The Jobs panel with one running job (with progress), one succeeded job, and one failed job with **Details** expanded. |
| `generate-with-grating.png` | operations-reference | The Generate form for a model with one grating added (Strength visible) and the seed field in sequence mode. |

---

## Priority 2

| File | Used in | Setup / what to capture |
|---|---|---|
| `startup-waiting.png` | getting-started | The "Waiting for backend server..." overlay. Start the app, then capture it before the backend is ready. |
| `new-project.png` | getting-started | The New Project panel with a name typed in and the "will be created at" preview showing. |
| `toolbar-file-menu.png` | interface | The File menu open, with 3 or more recent projects listed. |
| `toolbar-utilities.png` | interface | The Utilities menu open. |
| `node-types.png` | interface | One node of each type side by side: model, grating, audio (one favorited), image, latent, individual, group, source folder, room. A dedicated scratch project makes this easier. |
| `operations-palette.png` | interface | **Operations…** on an audio node, with "gen" typed into the search box. |
| `pick-mode.png` | interface | Graph-pick mode: the canvas dimmed, with the valid candidates outlined in green. |
| `ctrl-drag-group.png` | interface | Ctrl+drag of a node, with its ghost over a group that is highlighted green. |
| `element-details.png` | interface | The details panel for a generated audio node, with `context` expanded. |
| `audio-player.png` | interface | The player during playback, with the volume popover open. |
| `import-model.png` | workflows | Import Model on the **Shared Models** tab, with a model selected. |
| `generated-group.png` | workflows | A group from a seed sequence (e.g. `1..6`), with the auto labels (`seed: N`) visible. Turn on Detailed Labels. |
| `bend-panel.png` | workflows | The Configure Model Bending panel with a layer chosen and **Precompute Feature Clustering** ticked. |
| `lineage-exemplars.png` | operations-reference | Generate on a gen-2 or later individual, with 2–3 Lineage Exemplar presets listed. |
| `score-popper.png` | workflows | The Score popper open on an individual. |
| `room-inside.png` | workflows | The inside of a room: the faded doorway nodes and the breadcrumb `Project › Room`. |

---

## Nice to have (not linked yet)

- Spring edges and detailed labels turned on.
- Hide Evolutionary Edges turned on, with one individual selected so its edges reappear.
- A room seen from outside.
- Confirm Removal for a group, showing the "Keep child elements" checkbox.
