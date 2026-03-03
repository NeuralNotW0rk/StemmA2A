<!-- src/renderer/src/components/graph/GraphControls.svelte -->
<script lang="ts">
  import { selectionStore } from '../../utils/stores'

  interface Props {
    ontidy?: () => void
    onfit?: () => void
    oncenter?: () => void
  }

  let { ontidy, onfit, oncenter }: Props = $props()

  let isSelecting = $state(false)
  selectionStore.subscribe((store) => {
    isSelecting = store.isSelecting
  })
</script>

<div class="graph-controls">
  {#if isSelecting}
    <button
      class="cancel-selection"
      onclick={() => selectionStore.cancelSelection()}
      title="Cancel selection"
      aria-label="Cancel selection"
    >
      Cancel
    </button>
  {/if}
  <button onclick={() => ontidy?.()} title="Tidy layout" aria-label="Tidy layout"> Tidy </button>

  <button onclick={() => onfit?.()} title="Fit to screen" aria-label="Fit to screen"> Fit </button>

  <button onclick={() => oncenter?.()} title="Center view" aria-label="Center view">
    Center
  </button>
</div>

<style>
  .graph-controls {
    position: absolute;
    top: 1rem;
    right: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    z-index: 1000; /* High z-index */
  }

  .graph-controls button {
    background: var(--color-overlay-background-primary);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
    backdrop-filter: blur(10px);
  }

  .graph-controls button:hover {
    background: var(--color-overlay-background-secondary);
    border-color: var(--color-overlay-border-secondary);
    transform: translateY(-1px);
  }

  .graph-controls button.cancel-selection {
    background-color: var(--color-error);
  }
</style>
