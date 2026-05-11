<!-- src/renderer/src/components/views/ElementInfoView.svelte -->
<script lang="ts">
  import { ELEMENT_INFO_CONFIG } from '../../utils/app-config'
  import type { ElementData } from '../../utils/types'

  interface Props {
    selectedElementData: ElementData | null
  }

  let { selectedElementData }: Props = $props()

  let displayEntries = $derived(
    selectedElementData
      ? Object.entries(selectedElementData)
          .filter(([key]) => !(ELEMENT_INFO_CONFIG.ignoredKeys as Set<string>).has(key))
          .sort(([keyA], [keyB]) => {
            const indexA = (ELEMENT_INFO_CONFIG.priorityKeys as readonly string[]).indexOf(keyA)
            const indexB = (ELEMENT_INFO_CONFIG.priorityKeys as readonly string[]).indexOf(keyB)

            // If both are priority keys, preserve their order defined in the array
            if (indexA !== -1 && indexB !== -1) return indexA - indexB
            if (indexA !== -1) return -1 // Only A is a priority key
            if (indexB !== -1) return 1 // Only B is a priority key

            // Neither are priority keys, sort alphabetically
            return keyA.localeCompare(keyB)
          })
      : []
  )
</script>

<div class="view-content">
  {#if selectedElementData}
    <ul>
      {#each displayEntries as [key, value] (key)}
        <li>
          <strong class="key">{key}</strong>
          <span class="value">{JSON.stringify(value, null, 2)}</span>
        </li>
      {/each}
    </ul>
  {:else}
    <div class="content-empty">
      <p>Select an element on the graph to see its details.</p>
    </div>
  {/if}
</div>

<style>
  .view-content {
    padding: 1rem;
    height: 100%;
    overflow-y: auto;
  }
  .content-empty {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    text-align: center;
    color: var(--color-text-overlay-secondary);
  }

  ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  li {
    margin-bottom: 0.75rem;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
    word-wrap: break-word;
    display: flex;
    flex-direction: column;
  }

  .key {
    color: var(--color-text-overlay-secondary);
    font-weight: bold;
    margin-bottom: 0.25rem;
  }

  .value {
    color: var(--color-text-overlay-tertiary);
    white-space: pre-wrap;
    background-color: var(--color-background-glass-2);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
  }
</style>
