<!-- src/renderer/src/components/ErrorDialog.svelte -->
<script lang="ts">
  interface Props {
    show: boolean
    title: string
    message: string
    onclose: () => void
  }

  let { show, title, message, onclose }: Props = $props()

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      onclose()
    }
  }

  function handleOverlayClick(e: MouseEvent): void {
    if (e.target === e.currentTarget) onclose()
  }

  function handleOverlayKeydown(e: KeyboardEvent): void {
    if (e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) {
      onclose()
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if show}
  <div
    class="dialog-overlay"
    role="button"
    tabindex="0"
    aria-label="Close dialog"
    onclick={handleOverlayClick}
    onkeydown={handleOverlayKeydown}
  >
    <div
      class="dialog"
      role="dialog"
      aria-modal="true"
      aria-labelledby="dialog-title"
      tabindex="-1"
    >
      <div class="dialog-header">
        <h3 id="dialog-title">{title}</h3>
        <button onclick={onclose}>×</button>
      </div>

      <div class="dialog-content">
        <p>{message}</p>
      </div>

      <div class="dialog-actions">
        <button class="primary" onclick={onclose}>OK</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--color-overlay-background-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10001; /* Higher than ImportModelDialog */
    backdrop-filter: blur(4px);
  }

  .dialog {
    background: var(--color-background-medium);
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.5rem;
    min-width: 300px;
    max-width: 450px;
  }

  .dialog-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--color-border-glass-1);
    color: var(--color-error);
  }

  .dialog-header h3 {
    margin: 0;
  }

  .dialog-header button {
    background: none;
    border: none;
    color: var(--color-overlay-text);
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .dialog-content {
    padding: 1.5rem;
    color: var(--color-overlay-text);
  }

  .dialog-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--color-border-glass-1);
  }

  .dialog-actions button {
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .dialog-actions button.primary {
    background: var(--color-primary);
    border: 1px solid var(--color-primary);
    color: var(--color-overlay-text);
  }
</style>
