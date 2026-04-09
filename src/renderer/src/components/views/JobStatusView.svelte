<script lang="ts">
  import { jobStore } from '../../utils/job-management'
  import JobItem from '../JobItem.svelte'
</script>

<div class="job-status-view">
  {#if $jobStore.length === 0}
    <p class="no-jobs">No active jobs.</p>
  {:else}
    <div class="job-cards-container">
      {#each $jobStore as job (job.id)}
        <JobItem {job} />
      {/each}
    </div>
  {/if}
</div>

<style>
  .job-status-view {
    padding: 0.5rem;
    min-height: 5rem;
  }
  .no-jobs {
    text-align: center;
    color: var(--color-text-muted);
    font-size: 0.9rem;
    padding: 1rem 0;
  }
  .job-cards-container {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
</style>
