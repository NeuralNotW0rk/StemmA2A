<!-- src/renderer/src/components/graph/GraphControls.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();
</script>

<div class="graph-controls">
  <button on:click={() => dispatch('tidy')} title="Tidy layout" aria-label="Tidy layout">
    Tidy
  </button>

  <button on:click={() => dispatch('fit')} title="Fit to screen" aria-label="Fit to screen">
    Fit
  </button>

  <button on:click={() => dispatch('center')} title="Center view" aria-label="Center view">
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
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 0.5rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
    backdrop-filter: blur(10px);
  }

  .graph-controls button:hover {
    background: rgba(0, 0, 0, 0.8);
    border-color: rgba(255, 255, 255, 0.3);
    transform: translateY(-1px);
  }
</style>
