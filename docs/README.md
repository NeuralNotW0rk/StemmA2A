# StemmA2A documentation

StemmA2A is a desktop workbench for exploring generative models (Stable Audio and StyleGAN2) through a
node graph. You can generate audio or images, apply *gratings* that bend a model's internals, and
**evolve model modifications** with human-in-the-loop selection: you listen, score, and breed the
variants you like.

---

## For users

| Guide | What's in it |
|---|---|
| [Getting started](getting-started.md) | Install, first launch, creating a project |
| [Core concepts](concepts.md) | The mental model and glossary: graph, models, gratings, individuals, exemplars, rooms |
| [The interface](user-guide/interface.md) | Toolbar, graph canvas, node colors, mouse and keyboard, panels, jobs |
| [Workflows](user-guide/workflows.md) | Step-by-step: generating, bending, evolving, organizing, exporting |
| [Operations reference](user-guide/operations-reference.md) | Every operation and its parameters |

---

## For developers

| Guide | What's in it |
|---|---|
| [Architecture](developer/architecture.md) | Processes, local vs remote engine, parameter graph, storage, jobs |
| [Extending StemmA2A](developer/extending.md) | Adding operations, form-config schema, adding model adapters |
| [Remote GPU engine](developer/remote-engine.md) | Docker image, compose, Cloudflare tunnel, asset transfer |
| [Known issues](developer/known-issues.md) | Bugs and inconsistencies found while writing these docs |

Coding standards for contributors (Svelte 5 runes, strict TypeScript, fail-fast) are in
[`.agents/AGENTS.md`](../.agents/AGENTS.md).

---

## Screenshots

The images referenced in these docs live in [`images/`](images/). The list of shots still needed,
with what to capture for each one, is in [images/SHOTLIST.md](images/SHOTLIST.md).
