<!-- src/renderer/src/components/Sidebar.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  export let selectedNodeData: Record<string, any> | null = null;

  const dispatch = createEventDispatcher();

  // A list of keys to ignore for a cleaner display
  const ignoredKeys = new Set(['x', 'y', 'vx', 'vy', 'fx', 'fy', 'isExpanded', 'index']);
</script>

<aside class="sidebar">
  {#if selectedNodeData}
    <div class="sidebar-header">
      <h3>{selectedNodeData.alias || selectedNodeData.name || 'Element Details'}</h3>
      <button class="close-btn" on:click={() => dispatch('close')}>×</button>
    </div>
    <div class="sidebar-content">
      <ul>
        {#each Object.entries(selectedNodeData) as [key, value]}
          {#if !ignoredKeys.has(key)}
            <li>
              <strong class="key">{key}</strong>
              <span class="value">{JSON.stringify(value, null, 2)}</span>
            </li>
          {/if}
        {/each}
      </ul>
    </div>
  {:else}
    <div class="sidebar-content-empty">
      <p>Select an element on the graph to see its details.</p>
    </div>
  {/if}
</aside>

<style>
  .sidebar {
    position: absolute;
    z-index: 1000;
    top: 4.5rem; /* Space for the toolbar */
    left: 1rem;
    bottom: 1rem;
    width: 280px;
    
    display: flex;
    flex-direction: column;

    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 0.5rem;
    color: white;
    backdrop-filter: blur(10px);
    overflow: hidden; /* Ensures content respects the border radius */
  }

  .sidebar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    background-color: rgba(0, 0, 0, 0.2);
    flex-shrink: 0;
  }

  .close-btn {
    background: none;
    border: none;
    color: white;
    font-size: 1.5rem;
    line-height: 1;
    cursor: pointer;
    padding: 0 0.5rem;
    opacity: 0.7;
  }

  .close-btn:hover {
    opacity: 1;
  }

  .sidebar-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .sidebar-content {
    padding: 1rem;
    overflow-y: auto;
    flex-grow: 1;
  }

  .sidebar-content-empty {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    text-align: center;
    padding: 1rem;
    color: rgba(255, 255, 255, 0.5);
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
    color: rgba(255, 255, 255, 0.5);
    font-weight: bold;
    margin-bottom: 0.25rem;
  }

  .value {
    color: rgba(255, 255, 255, 0.9);
    white-space: pre-wrap;
    background-color: rgba(0, 0, 0, 0.2);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
  }

  /* Custom Scrollbar Styling */
  .sidebar-content::-webkit-scrollbar {
    width: 8px;
  }

  .sidebar-content::-webkit-scrollbar-track {
    background: transparent;
  }

  .sidebar-content::-webkit-scrollbar-thumb {
    background-color: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
  }

  .sidebar-content::-webkit-scrollbar-thumb:hover {
    background-color: rgba(255, 255, 255, 0.3);
  }
</style>