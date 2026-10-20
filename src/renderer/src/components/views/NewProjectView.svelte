<script lang="ts">
  interface Props {
    onclose: () => void
    oncreate: (data: { project_path: string; project_name: string }) => Promise<void>
  }

  import { onMount } from 'svelte'

  let { onclose, oncreate }: Props = $props()

  // Characters that aren't allowed in folder names on Windows (and '/' on all platforms)
  const INVALID_NAME_CHARS = /[<>:"/\\|?*]/

  let projectName = $state('')
  let parentLocation = $state<string | null>(null)
  let isLoading = $state(false)
  let errorMessage = $state<string | null>(null)

  let trimmedName = $derived(projectName.trim())
  let nameError = $derived.by(() => {
    if (INVALID_NAME_CHARS.test(trimmedName)) {
      return 'Name cannot contain any of: < > : " / \\ | ? *'
    }
    if (/[. ]$/.test(trimmedName)) {
      return 'Name cannot end with a period or space.'
    }
    return null
  })
  let projectPath = $derived.by(() => {
    if (!parentLocation || !trimmedName) return null
    const sep = parentLocation.includes('\\') ? '\\' : '/'
    return parentLocation.replace(/[/\\]+$/, '') + sep + trimmedName
  })
  let canCreate = $derived(!isLoading && !!projectPath && !nameError)

  onMount(async () => {
    try {
      parentLocation = await window.api.getDefaultProjectLocation()
    } catch (error) {
      console.error('Failed to get default project location:', error)
    }
  })

  async function selectProjectLocation(): Promise<void> {
    errorMessage = null
    try {
      const path = await window.api.newProject(parentLocation ?? undefined)
      if (path) {
        parentLocation = path
      }
    } catch (error: unknown) {
      errorMessage = error instanceof Error ? error.message : 'Failed to open directory dialog.'
    }
  }

  async function handleCreate(): Promise<void> {
    if (!canCreate || !projectPath) return

    isLoading = true
    errorMessage = null

    try {
      await oncreate({ project_path: projectPath, project_name: trimmedName })
      onclose() // Close panel on success
    } catch (error: unknown) {
      errorMessage =
        error instanceof Error
          ? error.message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '')
          : 'An unknown error occurred.'
    } finally {
      isLoading = false
    }
  }
</script>

<div class="new-project-view">
  <div class="form-item">
    <label for="project-name">Project Name</label>
    <!-- svelte-ignore a11y_autofocus -->
    <input
      id="project-name"
      type="text"
      bind:value={projectName}
      placeholder="e.g., My Project"
      disabled={isLoading}
      autofocus
      onkeydown={(e) => e.key === 'Enter' && handleCreate()}
    />
    {#if nameError}
      <span class="field-error">{nameError}</span>
    {/if}
  </div>

  <div class="form-item">
    <label for="project-location">Location</label>
    <div class="location-picker">
      <span class="path-display" title={parentLocation ?? ''}>
        {parentLocation || 'No location selected'}
      </span>
      <button
        id="project-location"
        onclick={selectProjectLocation}
        disabled={isLoading}
        class="secondary"
      >
        Browse...
      </button>
    </div>
  </div>

  {#if projectPath && !nameError}
    <p class="description">
      Project folder will be created at <span class="path-preview">{projectPath}</span>
    </p>
  {/if}

  {#if errorMessage}
    <div class="error-message">{errorMessage}</div>
  {/if}

  <div class="actions">
    <button onclick={onclose} disabled={isLoading} class="secondary"> Cancel </button>
    <button onclick={handleCreate} disabled={!canCreate} class="primary">
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
    font-size: 0.875rem;
    margin: 0;
    word-break: break-all;
  }
  .path-preview {
    color: var(--color-text-overlay-primary);
    font-family: monospace;
  }
  .field-error {
    color: var(--color-error);
    font-size: 0.8125rem;
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
    flex: 1;
    min-width: 0;
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
