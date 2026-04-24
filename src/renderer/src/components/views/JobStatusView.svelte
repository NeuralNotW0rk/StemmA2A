<script lang="ts">
  import { jobStore } from '../../utils/job-management'
  import JobItem from '../JobItem.svelte'
</script>

<div class="job-status-view">
  <div class="job-cards-container">
    {#each $jobStore as job (job.id)}
      <JobItem {job} />
    {/each}
  </div>
</div>

<style>
  .job-status-view {
    padding: 0.5rem;
    min-height: 5rem;
  }
  .job-cards-container {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
</style>
