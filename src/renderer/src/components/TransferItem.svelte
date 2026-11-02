<script lang="ts">
  import { transferDetail, transferPercent, type Transfer } from '../utils/transfers'

  let { transfer } = $props<{ transfer: Transfer }>()

  const percent = $derived(transferPercent(transfer))
</script>

<div
  class="transfer-item"
  class:failed={transfer.state === 'failed'}
  class:done={transfer.state === 'done'}
  title={transfer.uid}
>
  <div class="transfer-main-row">
    <span class="transfer-direction" aria-label={transfer.direction}>
      {transfer.direction === 'upload' ? '↑' : '↓'}
    </span>
    <span class="transfer-name">{transfer.name}</span>
    {#if percent !== null && transfer.state !== 'failed'}
      <span class="transfer-percent">{Math.floor(percent)}%</span>
    {/if}
  </div>
  {#if transfer.state === 'active' || transfer.state === 'queued'}
    <!-- Without a known size the bar is indeterminate -->
    <progress value={percent ?? undefined} max="100"></progress>
  {/if}
  <div class="transfer-detail">{transferDetail(transfer)}</div>
</div>

<style>
  .transfer-item {
    border: 1px solid var(--color-border-glass-1, var(--color-border));
    border-left: 4px solid var(--color-info, #3b82f6);
    border-radius: 0.5rem;
    padding: 0.5rem 0.65rem;
    background-color: var(--color-background-glass-2, var(--color-bg-2));
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .transfer-item.done {
    border-left-color: var(--color-success, #22c55e);
  }

  .transfer-item.failed {
    border-left-color: var(--color-error, #ef4444);
  }

  .transfer-main-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
  }

  .transfer-direction {
    font-weight: 700;
    color: var(--color-text-2);
  }

  .transfer-name {
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
    min-width: 0;
  }

  .transfer-percent {
    font-variant-numeric: tabular-nums;
    color: var(--color-text-2);
  }

  progress {
    width: 100%;
    margin: 0;
    height: 6px;
  }

  .transfer-detail {
    font-size: 0.72rem;
    color: var(--color-text-2);
    font-variant-numeric: tabular-nums;
    word-break: break-word;
  }
</style>
