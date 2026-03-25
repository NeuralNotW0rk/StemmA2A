<script lang="ts">
  import { executionStore, clearExecution } from '../../utils/execution'

  function close(): void {
    clearExecution()
  }
</script>

<div class="view-container">
  <div class="view-content">
    {#if $executionStore.status === 'running'}
      <div class="centered-content">
        <div class="spinner"></div>
        <p>Generation in progress...</p>
      </div>
    {:else if $executionStore.status === 'success'}
      <div class="centered-content">
        <p>Generation successful!</p>
        <!-- TODO: Display result details -->
        <pre>{JSON.stringify($executionStore.result, null, 2)}</pre>
      </div>
    {:else if $executionStore.status === 'error'}
      <div class="error-message">
        <h4>{$executionStore.error.title}</h4>
        <p>{$executionStore.error.message}</p>
      </div>
    {/if}
  </div>
  <div class="panel-actions">
    <button onclick={close} disabled={$executionStore.status === 'running'}>Close</button>
  </div>
</div>

<style>
  .view-container {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  .view-content {
    flex-grow: 1;
    overflow-y: auto;
    padding: 1rem;
  }
  .centered-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    height: 100%;
  }
  .error-message {
    padding: 1rem;
    color: var(--color-error);
  }
  .panel-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    padding: 1rem;
    border-top: 1px solid var(--color-border-glass-1);
    background-color: var(--color-background-glass-2);
    flex-shrink: 0;
  }
  pre {
    white-space: pre-wrap;
    word-break: break-all;
  }
</style>
