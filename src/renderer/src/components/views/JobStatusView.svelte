<script lang="ts">
  import { flip } from 'svelte/animate'
  import { jobStore, type Job, type JobStatus } from '../../utils/job-management'
  import JobItem from '../JobItem.svelte'
  import TransferItem from '../TransferItem.svelte'
  import { transferStore, isNotableTransfer } from '../../utils/transfers'

  const getStatusPriority = (status: JobStatus): number => {
    switch (status) {
      case 'running':
      case 'cancelling':
        return 0
      case 'pending':
        return 1
      case 'error':
      case 'success':
      case 'cancelled':
      default:
        return 2
    }
  }

  const sortedJobs = $derived<Job[]>(
    [...$jobStore].sort((a: Job, b: Job): number => {
      const priorityDiff = getStatusPriority(a.status) - getStatusPriority(b.status)
      if (priorityDiff !== 0) {
        return priorityDiff
      }
      // For active/pending jobs, maintain creation order (earlier created first)
      if (a.status === 'running' || a.status === 'pending' || a.status === 'cancelling') {
        return a.createdAt - b.createdAt
      }
      // For completed jobs, show most recently updated/completed first
      return b.updatedAt - a.updatedAt
    })
  )

  // Re-evaluated on every transfer poll, so a transfer appears once it has been running a while
  const notableTransfers = $derived($transferStore.filter(isNotableTransfer))
</script>

<div class="job-status-view">
  <div class="job-cards-container">
    {#each sortedJobs as job (job.id)}
      <div animate:flip={{ duration: 250 }}>
        <JobItem {job} />
      </div>
    {/each}
  </div>
  {#if notableTransfers.length > 0}
    <section class="transfers" aria-label="File transfers">
      <h4>Transfers</h4>
      {#each notableTransfers as transfer (transfer.id)}
        <TransferItem {transfer} />
      {/each}
    </section>
  {/if}
</div>

<style>
  .job-status-view {
    padding: 0.5rem;
    min-height: 5rem;
  }
  .transfers {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }
  .transfers h4 {
    margin: 0;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-text-2);
  }
  .job-cards-container {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
</style>
