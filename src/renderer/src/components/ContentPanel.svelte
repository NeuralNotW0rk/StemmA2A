<!-- src/renderer/src/components/ContentPanel.svelte -->
<script lang="ts">
  let {
    title,
    onclose,
    position = 'left'
  } = $props<{
    title: string
    onclose?: () => void
    position?: 'left' | 'right'
  }>()
</script>

<aside class="content-panel" class:left={position === 'left'} class:right={position === 'right'}>
  <div class="panel-header">
    <h3>{title}</h3>
    {#if onclose}
      <button class="close-btn" onclick={onclose}>×</button>
    {/if}
  </div>
  <div class="panel-body">
    <slot />
  </div>
</aside>

<style>
  .content-panel {
    position: absolute;
    z-index: 1000;
    top: 4.5rem;
    bottom: 1rem;
    display: flex;
    flex-direction: column;
    background: var(--color-background-glass-1);
    border: 1px solid var(--color-border-glass-1);
    border-radius: 0.5rem;
    color: var(--color-overlay-text);
    backdrop-filter: blur(10px);
    overflow: hidden;
  }

  .content-panel.left {
    left: 1rem;
    width: 320px;
  }

  .content-panel.right {
    right: 1rem;
    width: 320px;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 0.5rem 0.75rem 1rem;
    border-bottom: 1px solid var(--color-border-glass-1);
    background-color: var(--color-background-glass-2);
    flex-shrink: 0;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .close-btn {
    background: none;
    border: none;
    color: var(--color-overlay-text);
    font-size: 1.5rem;
    line-height: 1;
    cursor: pointer;
    padding: 0 0.5rem;
    opacity: 0.7;
    min-width: unset;
    min-height: unset;
  }

  .close-btn:hover {
    opacity: 1;
  }

  .panel-body {
    /* NO PADDING HERE */
    flex-grow: 1;
    min-height: 0;
  }

  /* Custom Scrollbar Styling */
  .panel-body::-webkit-scrollbar {
    width: 8px;
  }

  .panel-body::-webkit-scrollbar-track {
    background: transparent;
  }

  .panel-body::-webkit-scrollbar-thumb {
    background-color: var(--color-overlay-border-primary);
    border-radius: 4px;
  }

  .panel-body::-webkit-scrollbar-thumb:hover {
    background-color: var(--color-overlay-border-secondary);
  }
</style>
