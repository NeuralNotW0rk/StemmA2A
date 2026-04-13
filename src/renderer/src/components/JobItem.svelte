<script lang="ts">
  import { cancelJob, removeJob, type Job } from '../utils/job-management'
  import { cyInstanceStore } from '../utils/stores'
  import { slide } from 'svelte/transition'

  let { job } = $props<{ job: Job }>()

  let showErrorDetails = $state(false)

  function handleFocus(job: Job): void {
    console.log('[JobItem] handleFocus triggered for job:', job.id, job.name)
    const cy = $cyInstanceStore
    if (!cy) {
      console.warn('[JobItem] Cytoscape instance not found')
      return
    }
    if (!job.result) {
      console.warn('[JobItem] job.result is empty', job)
      return
    }

    console.log('[JobItem] Raw job.result:', job.result)
    // Extract target ID whether it is a string directly, inside an object, or in an array
    let targetId: string | number | undefined
    if (typeof job.result === 'string' || typeof job.result === 'number') {
      targetId = job.result
    } else if (Array.isArray(job.result) && job.result.length > 0) {
      const first = job.result[0]
      targetId = typeof first === 'object' ? first?.node_id || first?.id || first?.data?.id : first
    } else if (typeof job.result === 'object') {
      targetId =
        job.result.node_id ||
        job.result.id ||
        job.result.data?.id ||
        job.result.artifact?.node_id ||
        job.result.artifact?.id ||
        job.result.artifact?.data?.id
    }

    console.log('[JobItem] Extracted targetId:', targetId)

    if (targetId !== undefined && targetId !== null) {
      const ele = cy.$id(String(targetId))
      console.log(
        '[JobItem] Found element in graph:',
        ele && ele.length > 0 ? ele.data() : 'None found'
      )
      if (ele && ele.length > 0) {
        cy.animate({ center: { eles: ele }, zoom: 1.5 }, { duration: 400 })
        ele.emit('tap')
      } else {
        console.warn(`[JobItem] Element with ID ${targetId} not found in the active graph.`)
      }
    }
  }
</script>

<div
  class="job-item"
  class:success={job.status === 'success'}
  class:error={job.status === 'error'}
  class:pending={job.status === 'pending'}
>
  <div class="job-main-row">
    <span class="job-name" title={job.name}>{job.name}</span>

    {#if (job.status === 'running' || job.status === 'pending') && job.progress}
      <div class="progress-container" title={job.progress.description || ''}>
        <progress value={job.progress.value} max={job.progress.total}></progress>
      </div>
    {/if}

    <div class="job-actions">
      <span class="job-status job-status-{job.status}">{job.status}</span>

      {#if job.status === 'error'}
        <button
          onclick={(e) => {
            e.stopPropagation()
            showErrorDetails = !showErrorDetails
          }}
        >
          {showErrorDetails ? 'Hide' : 'Details'}
        </button>
        <button
          class="remove-btn icon-btn"
          title="Dismiss"
          onclick={(e) => {
            e.stopPropagation()
            removeJob(job.id)
          }}>✕</button
        >
      {:else if job.status === 'success'}
        <button
          onclick={(e) => {
            e.stopPropagation()
            handleFocus(job)
          }}>View</button
        >
        <button
          class="remove-btn icon-btn"
          title="Dismiss"
          onclick={(e) => {
            e.stopPropagation()
            removeJob(job.id)
          }}>✕</button
        >
      {:else if job.status === 'running' || job.status === 'cancelling' || job.status === 'pending'}
        <button
          onclick={(e) => {
            e.stopPropagation()
            cancelJob(job.id)
          }}
          disabled={job.status === 'cancelling'}
        >
          {job.status === 'cancelling'
            ? 'Stopping...'
            : job.status === 'pending'
              ? 'Cancel'
              : 'Stop'}
        </button>
      {:else}
        <button
          class="remove-btn icon-btn"
          title="Dismiss"
          onclick={(e) => {
            e.stopPropagation()
            removeJob(job.id)
          }}>✕</button
        >
      {/if}
    </div>
  </div>

  {#if job.status === 'error' && showErrorDetails && job.error}
    <div class="error-details" transition:slide>
      {#if typeof job.error === 'string'}
        <pre>{job.error}</pre>
      {:else}
        {#if job.error.title}
          <strong>{job.error.title}</strong>
        {/if}
        <pre>{job.error.message ||
            (job.error as { stack?: string }).stack ||
            JSON.stringify(job.error)}</pre>
      {/if}
    </div>
  {/if}
</div>

<style>
  .job-item {
    flex-shrink: 0;
    border: 1px solid var(--color-border-glass-1, var(--color-border));
    border-radius: 0.5rem;
    padding: 0.85rem;
    background-color: var(--color-background-glass-2, var(--color-bg-2));
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition:
      transform 0.2s ease,
      box-shadow 0.2s ease,
      background-color 0.2s ease;
    overflow: hidden;
  }

  .job-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
  }

  .job-item.success {
    border-left: 4px solid var(--color-success);
  }

  .job-item.error {
    border-left: 4px solid var(--color-error);
  }

  .job-item.pending {
    border-left: 4px solid var(--color-warning, #f59e0b);
  }

  .job-item.clickable {
    cursor: pointer;
  }

  .job-item.clickable:hover {
    background-color: var(--color-bg-3);
  }

  .job-main-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: space-between;
  }

  .job-name {
    font-weight: 600;
    font-size: 0.85rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 1;
    min-width: 0;
  }

  .job-status {
    font-size: 0.75rem;
    padding: 2px 6px;
    border-radius: 3px;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  .job-status-running,
  .job-status-cancelling {
    background-color: var(--color-info, #3b82f6);
    color: #fff;
  }
  .job-status-pending {
    background-color: var(--color-warning, #f59e0b);
    color: #fff;
  }
  .job-status-success {
    background-color: var(--color-success, #22c55e);
    color: #fff;
  }
  .job-status-error {
    background-color: var(--color-error, #ef4444);
    color: #fff;
  }
  .job-status-cancelled {
    background-color: var(--color-bg-3);
    color: var(--color-text-2);
  }

  .progress-container {
    flex-grow: 1;
    display: flex;
    align-items: center;
    min-width: 40px;
  }

  progress {
    width: 100%;
    margin: 0;
    height: 6px;
  }

  .job-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-shrink: 0;
  }

  .job-actions button {
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    min-width: auto;
    min-height: auto;
  }

  .job-actions button.icon-btn {
    padding: 0.25rem;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
  }

  .error-details {
    margin-top: 0.75rem;
    padding: 0.5rem;
    background-color: var(--color-bg-1);
    border-radius: 4px;
    border: 1px solid var(--color-border);
  }

  .error-details pre {
    white-space: pre-wrap;
    word-break: break-all;
    font-size: 0.8rem;
    color: var(--color-text-2);
    margin-top: 0.25rem;
  }
</style>
