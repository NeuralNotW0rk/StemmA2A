<script lang="ts">
  import { jobStore, removeJob, cancelJob } from '../../utils/job-management'
  import type { Job } from '../../utils/job-management'

  function getStatusClass(status: Job['status']): string {
    switch (status) {
      case 'running':
        return 'status-running'
      case 'success':
        return 'status-success'
      case 'error':
        return 'status-error'
      case 'cancelled':
      case 'cancelling':
        return 'status-cancelled'
      default:
        return ''
    }
  }
</script>

<div class="job-status-view">
  {#if $jobStore.length === 0}
    <p class="no-jobs">No active jobs.</p>
  {:else}
    <ul>
      {#each $jobStore as job (job.id)}
        <li class="job-item">
          <span class="job-name">{job.name}</span>
          <div class="job-details">
            <span class="job-status {getStatusClass(job.status)}">{job.status}</span>
            {#if job.status === 'running' || job.status === 'cancelling'}
              <button
                class="cancel-btn"
                onclick={() => cancelJob(job.id)}
                disabled={job.status === 'cancelling'}>Cancel</button
              >
            {:else}
              <button class="close-btn" onclick={() => removeJob(job.id)}>×</button>
            {/if}
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .job-status-view {
    padding: 0.5rem;
  }
  .no-jobs {
    text-align: center;
    color: var(--color-text-muted);
    font-size: 0.9rem;
    padding: 1rem 0;
  }
  ul {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .job-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: var(--color-background-glass-2);
    padding: 0.5rem 0.75rem;
    border-radius: 0.25rem;
    border: 1px solid var(--color-border-glass-1);
  }
  .job-name {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.9rem;
  }
  .job-details {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .job-status {
    font-size: 0.8rem;
    padding: 0.1rem 0.4rem;
    border-radius: 0.25rem;
    text-transform: uppercase;
  }
  .status-running {
    background-color: #3b82f6;
    color: white;
  }
  .status-success {
    background-color: #22c55e;
    color: white;
  }
  .status-error {
    background-color: #ef4444;
    color: white;
  }
  .status-cancelled {
    background-color: #71717a;
    color: white;
  }
  .close-btn {
    background: none;
    border: none;
    color: var(--color-overlay-text);
    font-size: 1.2rem;
    line-height: 1;
    cursor: pointer;
    padding: 0 0.25rem;
    opacity: 0.7;
    min-width: unset;
    min-height: unset;
  }
  .close-btn:hover {
    opacity: 1;
  }
  .cancel-btn {
    font-size: 0.8rem;
    background-color: var(--color-background-raised);
    border: 1px solid var(--color-border);
    color: var(--color-text);
    padding: 0.2rem 0.5rem;
    border-radius: 0.25rem;
    cursor: pointer;
  }
  .cancel-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
