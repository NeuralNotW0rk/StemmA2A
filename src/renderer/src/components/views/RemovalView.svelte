<!-- c:\Users\griff\Documents\GitHub\StemmA2A\src\renderer\src\components\views\RemovalView.svelte -->
<script lang="ts">
  import { selectedForRemoval } from '../../utils/stores'

  let { onclose, onrefresh, onerror } = $props<{
    onclose: () => void
    onrefresh: () => void
    onerror: (error: { title: string; message: string }) => void
  }>()

  let isRemoving = $state(false)

  async function remove(): Promise<void> {
    if (!$selectedForRemoval) return

    isRemoving = true
    try {
      await window.api.removeElement($selectedForRemoval.id)
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
  {#if $selectedForRemoval}
    <h2>Confirm Removal</h2>
    <p>Are you sure you want to remove the following element?</p>
    <div class="element-info">
      <strong>{$selectedForRemoval.id}</strong>
      <span>({$selectedForRemoval.type})</span>
    </div>
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
