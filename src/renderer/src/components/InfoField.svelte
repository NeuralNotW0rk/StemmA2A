<!-- src/renderer/src/components/InfoField.svelte -->
<script lang="ts">
  import { ELEMENT_INFO_CONFIG } from '../utils/app-config'
  import InfoField from './InfoField.svelte'

  interface Props {
    name: string
    value: unknown
    path: string
    isExpanded: (path: string, value: unknown) => boolean
    toggle: (path: string, value: unknown) => void
  }

  let { name, value, path, isExpanded, toggle }: Props = $props()

  function isPrimitive(v: unknown): boolean {
    return v === null || typeof v !== 'object'
  }

  // Objects and arrays of non-primitives are rendered as nested fields;
  // primitives and flat arrays (e.g. embedding vectors) are rendered as a single value
  let isContainer = $derived(
    !isPrimitive(value) && !(Array.isArray(value) && value.every(isPrimitive))
  )
  let children = $derived(isContainer ? Object.entries(value as object) : [])
  let expanded = $derived(isExpanded(path, value))

  function formatValue(v: unknown): string {
    if (Array.isArray(v)) return JSON.stringify(v)
    return JSON.stringify(v, null, 2) ?? String(v)
  }

  function summarize(v: unknown): string {
    if (Array.isArray(v)) return `Array(${v.length})`
    if (v !== null && typeof v === 'object') {
      const count = Object.keys(v).length
      return `{${count} ${count === 1 ? 'key' : 'keys'}}`
    }
    const text = JSON.stringify(v) ?? String(v)
    return text.length > ELEMENT_INFO_CONFIG.previewLength
      ? `${text.slice(0, ELEMENT_INFO_CONFIG.previewLength)}…`
      : text
  }
</script>

<li>
  <button
    class="key-toggle"
    onclick={() => toggle(path, value)}
    aria-expanded={expanded}
    type="button"
  >
    <span class="chevron" class:open={expanded}>▸</span>
    <strong class="key">{name}</strong>
    {#if !expanded || isContainer}
      <span class="preview">{summarize(value)}</span>
    {/if}
  </button>
  {#if expanded}
    {#if isContainer}
      <ul class="children">
        {#each children as [childKey, childValue] (childKey)}
          <InfoField
            name={childKey}
            value={childValue}
            path={`${path}/${childKey}`}
            {isExpanded}
            {toggle}
          />
        {/each}
      </ul>
    {:else}
      <span class="value">{formatValue(value)}</span>
    {/if}
  {/if}
</li>

<style>
  li {
    margin-bottom: 0.75rem;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
    word-wrap: break-word;
    display: flex;
    flex-direction: column;
  }

  .children {
    list-style: none;
    margin: 0.25rem 0 0 0.3rem;
    padding: 0 0 0 0.75rem;
    border-left: 1px solid var(--color-border-overlay-subtle, rgba(255, 255, 255, 0.1));
  }

  .children > :global(li) {
    margin-bottom: 0.5rem;
  }

  .key-toggle {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    min-width: 0;
    background: none;
    border: none;
    padding: 0;
    margin-bottom: 0.25rem;
    cursor: pointer;
    font: inherit;
    text-align: left;
    user-select: none;
  }

  .chevron {
    flex-shrink: 0;
    color: var(--color-text-overlay-tertiary);
    transition: transform 0.15s;
  }

  .chevron.open {
    transform: rotate(90deg);
  }

  .key {
    color: var(--color-text-overlay-secondary);
    font-weight: bold;
  }

  .key-toggle:hover .key {
    color: var(--color-text-overlay);
  }

  .preview {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--color-text-overlay-tertiary);
    opacity: 0.7;
  }

  .value {
    color: var(--color-text-overlay-tertiary);
    white-space: pre-wrap;
    background-color: var(--color-background-glass-2);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    cursor: text;
  }
</style>
