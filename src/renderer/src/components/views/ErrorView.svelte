<!-- src/renderer/src/components/views/ErrorView.svelte -->
<script lang="ts">
  let { title, message } = $props<{
    title: string
    message: string
  }>()
</script>

<div class="error-view">
  <h4>{title}</h4>
  <p>{message}</p>
</div>

<style>
  .error-view {
    padding: 1rem;
    color: var(--color-error);
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    overflow-y: auto;
  }
  h4 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
  }
  p {
    margin: 0;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.9rem;
    white-space: pre-wrap;
    word-wrap: break-word;
    background-color: var(--color-background-glass-2);
    padding: 0.5rem;
    border-radius: 0.25rem;
  }
</style>
