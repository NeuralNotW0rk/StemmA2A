<!-- c:\Users\griff\Documents\GitHub\StemmA2A\src\renderer\src\components\views\RemovalView.svelte -->
<script lang="ts">
  import { selectedForRemoval } from '../../utils/stores'
  import type { ErrorInfo } from '../../utils/types'

  let { onclose, onrefresh, onerror } = $props<{
    onclose: () => void
    onrefresh: () => void
    onerror: (error: ErrorInfo) => void
  }>()

  let isRemoving = $state(false)
  let keepChildren = $state(true)

  let elements = $derived(
    Array.isArray($selectedForRemoval)
      ? $selectedForRemoval
      : [$selectedForRemoval].filter(Boolean)
  )

  async function remove(): Promise<void> {
    if (elements.length === 0) return

    isRemoving = true
    try {
      const ids = elements.map(el => el.id)
      await window.api.removeElements(ids, keepChildren)
      onrefresh()
    } catch (e: unknown) {
      console.error('Failed to remove element:', e)
      const message = e instanceof Error ? e.message : String(e)
      onerror({ title: 'Removal Failed', message })
    } finally {
      isRemoving = false
    }
  }
</script>

<div class="view-content">
  {#if elements.length > 0}
    <h2>Confirm Removal</h2>
    <p>
      Are you sure you want to remove 
      {#if elements.length > 1}
        these {elements.length} elements?
      {:else}
        the following element?
      {/if}
    </p>
    
    {#if elements.length === 1}
      <div class="element-info">
        <strong>{elements[0].id}</strong>
        <span>({elements[0].type})</span>
      </div>
    {/if}

    {#if elements.some(el => el.type === 'batch' || el.type === 'directory')}
      <div class="options">
        <label>
          <input type="checkbox" bind:checked={keepChildren} disabled={isRemoving} />
          {#if elements.some(el => el.type === 'batch')}
            Keep child elements (disband batch)
          {:else}
            Keep contained files
          {/if}
        </label>
        <p class="hint-text">
          {#if keepChildren}
            The container elements will be removed, but their contents will remain on the graph.
          {:else}
            The container elements and all of their contents will be permanently removed.
          {/if}
        </p>
      </div>
    {/if}
    <div class="button-group">
      <button onclick={remove} class="danger" disabled={isRemoving}>
        {#if isRemoving}
          <div class="spinner"></div>
        {:else}
          Remove
        {/if}
      </button>
      <button onclick={onclose} disabled={isRemoving}>Cancel</button>
    </div>
  {:else}
    <div class="content-empty">
      <p>No element selected for removal.</p>
    </div>
  {/if}
</div>

<style>
  .view-content {
    padding: 1rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }

  .element-info {
    margin: 1rem 0;
    padding: 0.5rem;
    background-color: var(--color-background-glass-2);
    border-radius: 0.25rem;
  }

  .options {
    margin-bottom: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    text-align: left;
  }

  .hint-text {
    font-size: 0.85rem;
    margin: 0;
    color: var(--color-text-dimmed, #aaa);
    max-width: 250px;
    line-height: 1.3;
  }

  .button-group {
    display: flex;
    gap: 0.5rem;
  }

  button {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
  }

  button.danger {
    background-color: var(--color-accent-danger);
    color: var(--color-text-button);
  }
</style>
