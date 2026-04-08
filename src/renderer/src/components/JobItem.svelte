<script lang="ts">
  import { focusJobNode, cancelJob, removeJob, type Job } from '../utils/job-management'
  import { slide } from 'svelte/transition'

  export let job: Job

  let showErrorDetails = false
</script>

<div class="job-item" class:success={job.status === 'success'} class:error={job.status === 'error'}>
  <div class="job-header">
    <span class="job-name">{job.name}</span>
    <span class="job-status job-status-{job.status}">{job.status}</span>
  </div>

  {#if job.status === 'running' && job.progress}
    <div class="progress-container" transition:slide={{ duration: 200 }}>
      <progress value={job.progress.value} max={job.progress.total}></progress>
      {#if job.progress.description}
        <span class="progress-description">{job.progress.description}</span>
      {/if}
    </div>
  {/if}

  {#if job.status === 'error'}
    <div class="job-actions">
      <button on:click={() => (showErrorDetails = !showErrorDetails)}>
        {showErrorDetails ? 'Hide' : 'Show'} Details
      </button>
      <button class="remove-btn" on:click={() => removeJob(job.id)}>Dismiss</button>
    </div>
    {#if showErrorDetails && job.error}
      <div class="error-details" transition:slide>
        <strong>{job.error.title}</strong>
        <pre>{job.error.message}</pre>
      </div>
    {/if}
  {:else if job.status === 'success'}
    <div class="job-actions">
      {#if job.result?.node_id || job.result?.id}
        <button on:click={() => focusJobNode(job)}>View in Graph</button>
      {/if}
      <button class="remove-btn" on:click={() => removeJob(job.id)}>Dismiss</button>
    </div>
  {:else if job.status === 'running' || job.status === 'cancelling'}
    <div class="job-actions">
      <button on:click={() => cancelJob(job.id)} disabled={job.status === 'cancelling'}>
        {job.status === 'cancelling' ? 'Cancelling...' : 'Cancel'}
      </button>
    </div>
  {:else}
    <div class="job-actions">
      <button class="remove-btn" on:click={() => removeJob(job.id)}>Dismiss</button>
    </div>
  {/if}
</div>

<style>
  .job-item {
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    background-color: var(--color-bg-2);
    transition: background-color 0.2s ease;
    overflow: hidden;
  }

  .job-item.success {
    border-left: 3px solid var(--color-success);
  }

  .job-item.error {
    border-left: 3px solid var(--color-error);
  }

  .job-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .job-name {
    font-weight: 600;
  }

  .job-status {
    font-size: 0.8rem;
    padding: 2px 6px;
    border-radius: 10px;
    text-transform: capitalize;
  }

  .job-status-running,
  .job-status-cancelling {
    background-color: var(--color-info-muted);
    color: var(--color-info);
  }
  .job-status-success {
    background-color: var(--color-success-muted);
    color: var(--color-success);
  }
  .job-status-error {
    background-color: var(--color-error-muted);
    color: var(--color-error);
  }
  .job-status-cancelled {
    background-color: var(--color-bg-3);
    color: var(--color-text-2);
  }

  .progress-container {
    margin-bottom: 0.5rem;
  }

  progress {
    width: 100%;
  }

  .progress-description {
    font-size: 0.8rem;
    color: var(--color-text-2);
  }

  .job-actions {
    margin-top: 0.5rem;
    display: flex;
    gap: 0.5rem;
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