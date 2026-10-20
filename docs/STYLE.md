# Documentation style

StemmA2A, [Diffracture](https://github.com/NeuralNotW0rk/Diffracture) and
[Neutral Selection](https://github.com/NeuralNotW0rk/NeutralSelection) share a light set of
documentation conventions, so the three projects read as a family. Diffracture's README is the
reference example. These are defaults, not rules: break them when a page reads better without them.

---

## README shape

Use the sections in this order, skipping any that don't apply:

1. `# Name` on the first line.
2. One paragraph that opens with the name in bold and says what the project is.
3. `> **Status:** …`: maturity and version.
4. `---`, then `## Contents` for any page with more than about four sections.
5. Installation, then Core concepts, then Quickstart or Usage.
6. Reference sections.
7. Extending.
8. Scripts and tests.
9. Known limitations.
10. Related projects.
11. Acknowledgements and prior work.
12. License.

---

## Formatting

- **Headings**: sentence case, for example "Core concepts" rather than "Core Concepts". Proper nouns
  and code identifiers keep their case.
- **Section breaks**: put `---` before each `##` section in READMEs and long reference pages. Shorter
  guide pages can do without.
- **Lists**: use `-` for bullets. When a bullet introduces a term, write it as
  `**Term**: explanation`.
- **Tables**: use plain `|---|` separators, without alignment colons. Use a table for anything with
  two or more attributes per row.
- **Code**: always give the fence a language (`bash`, `powershell`, `python`, `json`). Put
  identifiers, paths and config keys in backticks.
- **Callouts**: use blockquotes with a bold label, such as `> **Status:**`, `> **Note:**` or
  `> **Warning:**`. Use them sparingly.
- **Citations**: put each one in a blockquote: authors, *Title*, venue and year, then a link.
- **Emoji**: don't use them in headings or bullets.

---

## Voice and wording

- Use American spelling (*color*, *organize*, *normalize*), to match the code and the UI.
- Write in plain declarative sentences, in the present tense. Address the reader as "you" in guides.
- Name UI elements exactly as they appear on screen, in bold: **Scatter**, **+ Import Model**.
- Shared terms mean the same thing in all three projects:

  | Term | Defined by |
  |---|---|
  | Grating, Element, Kernel, Actant | Diffracture |
  | Genome, Segment, granularity, crossover, mutation | Neutral Selection |
  | Individual, exemplar, scatter, recombine, room | StemmA2A |

  Use these terms with the same meaning everywhere. Avoid near-synonyms such as "patch" for grating.
