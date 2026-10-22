<!-- src/renderer/src/components/views/MessageView.svelte -->
<script lang="ts">
  import type { PanelMessage } from '../../utils/types'

  let {
    title,
    message,
    kind = 'error'
  } = $props<{
    title: string
    message: string
    kind?: PanelMessage['kind']
  }>()
</script>

<div class="message-view {kind}">
  <h4>{title}</h4>
  <p>{message}</p>
</div>

<style>
  .message-view {
    padding: 1rem;
    color: var(--color-error);
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    overflow-y: auto;
  }
  .message-view.success {
    color: var(--color-success);
  }
  .message-view.info {
    color: var(--color-text);
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
