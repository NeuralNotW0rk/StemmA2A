# Core concepts

---

## The big idea

Most generative tools let you vary the **inputs** to a model: prompts, seeds, settings.
StemmA2A also lets you vary the **model itself**. It does this in two ways:

- **Bending**: you hand-pick a layer and a transform, such as eroding, rotating or ablating that
  layer's activations. The result is a *grating* you can apply to any generation.
- **Evolution**: you wrap something you like as an *individual*. An individual carries a small,
  low-rank modification (LoRA) for every layer of the model. You then *scatter* mutated children
  from it or *recombine* several parents. You listen to the results, score them, and repeat.
  Selection is done by you, not by an automatic fitness function.

Everything you make is a node in one **parameter graph**. The edges between nodes record where each
thing came from.

![A project graph with models, generated audio, a grating, and an evolved lineage](images/overview.png)

---

## Glossary

### Project and graph

**Project**
A folder containing `graph.json` (the graph), `embeddings.safetensors` (similarity vectors),
`generate/` and `process/` (output files with readable names), and `cache/` (content-addressed copies).

**Parameter graph**
The node graph shown on the canvas. Every model, artifact, collection and relationship in the
project lives here.

**Room**
A sub-view of the graph that you create with **Break Out**. Nothing is copied: nodes inside a room
are still linked to everything outside it. From outside, the room is a box labeled `Name · N inside`.
Inside, neighbours from outside show as faded *doorway* nodes.

**Fork**
**Fork to Project** copies the selected nodes, plus everything they depend on, into a new sibling
project. The new project remembers where it came from, and the toolbar shows a **←** button to return
to the original project.

### Models and modifications

**Model**
An imported generator. Two adapters are supported: **Stable Audio Tools**, which produces audio, and
**StyleGAN2**, which produces images.

**Shared model**
A model listed in `backend/shared_models.json`. On a remote engine it is listed on the server.
Shared models can be imported into any project without browsing for files.

**Grating**
A set of interventions on a model's layers, stored as a `.safetensors` file. A grating is applied at
generation time with a **strength**: 1 is normal, a negative value inverts the effect, and a value
above 1 exaggerates it. Gratings come from four places: you create them with **Bend**, you import
them, *express* them from an individual, or GAN inversion produces them. The term comes from the sibling
`diffracture` library.

**Bending kernel**
The transform a bending grating applies to a layer's activations. The options are Erode, Dilate,
Scale, Rotate, Translate, Scalar Multiply and Ablate. A kernel can target all channels, specific
channels, or channel clusters found automatically.

**Latent**
A starting point in the model's latent space, produced by **inversion** of an existing audio clip or
image. You can generate from a latent to reproduce or vary the original.

### Artifacts and collections

**Artifact**
Any generated or imported item: audio, image, latent, grating or individual.
Artifacts are **content-addressed**: their ID is a hash of their content, so two identical outputs
are the same node.

**Group**
A collection of artifacts of the same type, such as the outputs of one batch or one evolutionary
generation. A group is labeled automatically with its shared prompt. Each member is labeled with
what makes it different from the others, for example `seed: 1001`.

**Source / directory**
An external folder added with **Add Source**. It appears as a blue folder node, and **Expand** turns it
into a directory box holding its files.

**Spring edge**
A hidden-by-default edge that links artifacts which *sound or look similar*. Similarity is measured
with CLAP embeddings for audio and CLIP embeddings for images. Spring edges can carry a short label
describing the change from one artifact to the other.

### Evolution

**Individual**
One candidate in an evolutionary run. It holds a **genome**: one low-rank (LoRA) perturbation per
eligible layer of the base model, each of which can be switched on or off. It appears on the graph as
an orange dashed box labeled `name [fit: X]`.

**Generation 0 / precursor**
You start evolution by wrapping an artifact you like, the **precursor**, as an individual. The
resulting generation-0 genome is a no-op, so the individual sounds exactly like the unmodified model.
Variation comes from scattering.

**Exemplar**
An audio clip or image generated *with* an individual's modification applied. Exemplars are how you
hear or see an individual. They sit inside the individual's box. By default, children reuse their
parent's exemplar settings (prompt, seed and so on), so you can compare like for like.

**Fitness**
A score you set by hand with **Score** on an individual. Recombination prefers parents with higher
fitness.

**Scatter**
Makes N mutated children from one or more seed individuals. Scatter mutates only; it does not cross
genomes. Use it to explore around something you like.

**Recombine**
Breeds a new generation from scored parents: it selects parents (by tournament or at random), crosses
their genomes over, then applies light mutation.

**Express**
Writes an individual's modification out as a standalone grating file.

**Phenotypic duplicate**
A child whose exemplar is byte-for-byte identical to its ancestor's. It is marked with a dotted border
and a **[Duplicate]** tag.

---

## How the pieces connect

```
Model ──generate──▶ Audio/Image ──wrap──▶ Individual (gen 0)
  ▲                                           │
  │ binds_to                               scatter / recombine
Grating ◀──express── Individual (gen N) ◀─────┘
                         │
                         └─ exemplars (audio/image generated with the genome applied)
```
