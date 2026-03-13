<script lang="ts">
  interface Props {
    onclose: () => void
    oncreate: (data: { project_path: string; project_name: string }) => Promise<void>
  }

  let { onclose, oncreate }: Props = $props()

  let projectName = $state('')
  let projectPath = $state<string | null>(null)
  let isLoading = $state(false)
  let errorMessage = $state<string | null>(null)

  async function selectProjectLocation(): Promise<void> {
    errorMessage = null
    try {
      const path = await window.api.newProject()
      if (path) {
        projectPath = path
        projectName = path.split(/[/\\]/).pop() || ''
      }
    } catch (error: any) {
      errorMessage = error.message || 'Failed to open directory dialog.'
    }
  }

  async function handleCreate(): Promise<void> {
    if (!projectPath) {
      errorMessage = 'Please select a project location.'
      return
    }
    if (!projectName.trim()) {
      errorMessage = 'Project name cannot be empty.'
      return
    }

    isLoading = true
    errorMessage = null

    try {
      await oncreate({ project_path: projectPath, project_name: projectName.trim() })
      onclose() // Close panel on success
    } catch (error: any) {
      errorMessage = error.message || 'An unknown error occurred.'
    } finally {
      isLoading = false
    }
  }
</script>

<div class="new-project-view">
  <p class="description">Select a location for your new project folder.</p>

  <div class="form-item">
    <label for="project-location">Project Location</label>
    <div class="location-picker">
      <button onclick={selectProjectLocation} class="secondary"> Select Location... </button>
      <span class="path-display">{projectPath || 'No location selected'}</span>
    </div>
  </div>

  {#if projectPath}
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
  {/if}

  {#if errorMessage}
    <div class="error-message">{errorMessage}</div>
  {/if}

  <div class="actions">
    <button onclick={onclose} disabled={isLoading} class="secondary"> Cancel </button>
    <button
      onclick={handleCreate}
      disabled={isLoading || !projectName.trim() || !projectPath}
      class="primary"
    >
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
  .location-picker {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .path-display {
    font-style: italic;
    color: var(--color-text-overlay-secondary);
    font-size: 0.875rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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
