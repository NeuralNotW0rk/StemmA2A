<!-- src/renderer/src/components/views/ElementInfoView.svelte -->
<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  interface Props {
    selectedNodeData: Record<string, any> | null
  }

  let { selectedNodeData }: Props = $props()

  // A list of keys to ignore for a cleaner display
  const ignoredKeys = new Set(['x', 'y', 'vx', 'vy', 'fx', 'fy', 'isExpanded', 'index'])
</script>

<div class="view-content">
  {#if selectedNodeData}
    <ul>
      {#each Object.entries(selectedNodeData) as [key, value] (key)}
        {#if !ignoredKeys.has(key)}
          <li>
            <strong class="key">{key}</strong>
            <span class="value">{JSON.stringify(value, null, 2)}</span>
          </li>
        {/if}
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
