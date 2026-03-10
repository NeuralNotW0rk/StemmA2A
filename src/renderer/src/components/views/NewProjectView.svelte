<script lang="ts">
  interface Props {
    onclose: () => void
    oncreate: (data: { project_name: string }) => Promise<void>
  }

  let { onclose, oncreate }: Props = $props()

  let projectName = $state('')
  let isLoading = $state(false)
  let errorMessage = $state<string | null>(null)

  async function handleCreate(): Promise<void> {
    if (!projectName.trim()) {
      errorMessage = 'Project name cannot be empty.'
      return
    }

    isLoading = true
    errorMessage = null

    try {
      await oncreate({ project_name: projectName.trim() })
      onclose() // Close panel on success
    } catch (error: any) {
      errorMessage = error.message || 'An unknown error occurred.'
    } finally {
      isLoading = false
    }
  }
</script>

<div class="new-project-view">
  <p class="description">Enter a name for your new project.</p>

  <div class="form-item">
    <label for="project-name">Project Name</label>
    <input
      id="project-name"
      type="text"
      bind:value={projectName}
      placeholder="e.g., My Project"
      disabled={isLoading}
      onkeydown={(e) => e.key === 'Enter' && handleCreate()}
    />
  </div>

  {#if errorMessage}
    <div class="error-message">{errorMessage}</div>
  {/if}

  <div class="actions">
    <button onclick={onclose} disabled={isLoading} class="secondary"> Cancel </button>
    <button onclick={handleCreate} disabled={isLoading || !projectName.trim()} class="primary">
      {#if isLoading}
        <div class="spinner"></div>
      {:else}
        Create Project
      {/if}
    </button>
  </div>
</div>

<style>
  .new-project-view {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    padding: 0.5rem;
  }
  .description {
    color: var(--color-text-overlay-secondary);
  }
  .form-item {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  label {
    font-weight: 600;
    color: var(--color-text-overlay-primary);
  }
  input {
    padding: 0.75rem;
    background: var(--color-background-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.375rem;
    color: var(--color-overlay-text);
  }
  input:focus {
    outline: none;
    border-color: var(--color-primary);
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    margin-top: 1rem;
  }
  .error-message {
    color: var(--color-error);
    background-color: var(--color-error-t-10);
    border: 1px solid var(--color-error-t-50);
    padding: 0.75rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
  }
  .secondary {
    background: transparent;
  }
</style>
