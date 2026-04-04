<script lang="ts">
  import { onMount } from 'svelte'
  import { isCreatingNewProject, cyInstanceStore } from '../utils/stores'

  interface Props {
    currentProject?: string | null
    viewMode?: 'batch' | 'cluster'
    onprojectLoad?: (data: { project_path?: string; project_name?: string }) => void
    onviewModeChange?: (mode: 'batch' | 'cluster') => void
    onrefresh?: () => void
    onaddExternalSource?: () => void
    onimportModel?: () => void
    onupdateEmbeddings?: () => void
  }

  let {
    currentProject = null,
    viewMode = 'batch',
    onprojectLoad,
    onviewModeChange,
    onrefresh,
    onaddExternalSource,
    onimportModel,
    onupdateEmbeddings
  }: Props = $props()

  let showFileMenu = $state(false)
  let fileMenuElement: HTMLElement | undefined = $state()
  let recentProjects: string[] = $state([])
  let showUtilitiesMenu = $state(false)
  let utilitiesMenuElement: HTMLElement | undefined = $state()
  let springsVisible = $state(true)

  async function refreshRecentProjects(): Promise<void> {
    try {
      recentProjects = await window.api.getRecentProjects()
    } catch (error) {
      console.error('Failed to get recent projects:', error)
    }
  }

  onMount(async () => {
    await refreshRecentProjects()
  })

  async function openProject(): Promise<void> {
    showFileMenu = false
    const projectPath = await window.api.openProject()
    if (projectPath) {
      onprojectLoad?.({ project_path: projectPath })
    }
  }

  function loadRecentProject(path: string): void {
    showFileMenu = false
    onprojectLoad?.({ project_path: path })
  }

  async function handleRemoveRecent(path: string, event: MouseEvent): Promise<void> {
    event.stopPropagation() // Prevent the dropdown item from being clicked
    await window.api.removeRecentProject(path)
    await refreshRecentProjects()
  }

  function toggleViewMode(): void {
    const newMode = viewMode === 'batch' ? 'cluster' : 'batch'
    onviewModeChange?.(newMode)
  }

  function toggleSprings(): void {
    springsVisible = !springsVisible
    if ($cyInstanceStore) {
      if (springsVisible) {
        $cyInstanceStore.edges('[type="spring"]').removeClass('hidden')
      } else {
        $cyInstanceStore.edges('[type="spring"]').addClass('hidden')
      }
    }
    showUtilitiesMenu = false
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      showFileMenu = false
      showUtilitiesMenu = false
    }
  }

  export { refreshRecentProjects }
</script>

<svelte:window
  onkeydown={handleKeydown}
  onclick={(event: MouseEvent): void => {
    if (
      fileMenuElement &&
      event.target instanceof Node &&
      !fileMenuElement.contains(event.target)
    ) {
      showFileMenu = false
    }
      if (
        utilitiesMenuElement &&
        event.target instanceof Node &&
        !utilitiesMenuElement.contains(event.target)
      ) {
        showUtilitiesMenu = false
      }
  }}
/>

<div class="toolbar">
  <div class="toolbar-section">
    <div class="file-menu" bind:this={fileMenuElement}>
      <button
        class="project-button"
        onclick={() => (showFileMenu = !showFileMenu)}
        aria-label="Open file menu"
      >
        File
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="currentColor"
          class:rotate-180={showFileMenu}
          class="chevron"
        >
          <path d="M7 10l5 5 5-5z" />
        </svg>
      </button>

      {#if showFileMenu}
        <div class="project-dropdown">
          <button
            class="dropdown-item"
            onclick={() => {
              $isCreatingNewProject = true
              showFileMenu = false
            }}
          >
            New Project...
          </button>
          <button class="dropdown-item" onclick={openProject}> Open Project... </button>
          <div class="dropdown-divider"></div>
          <div class="dropdown-header">Recent Projects</div>
          {#each recentProjects as project (project)}
            <div class="dropdown-item-container">
              <button class="dropdown-item" onclick={() => loadRecentProject(project)}>
                {project.split(/[/\\]/).pop()}
                <span class="recent-project-path">{project}</span>
              </button>
              <button
                class="remove-recent"
                title="Remove from recent"
                onclick={(e) => handleRemoveRecent(project, e)}
              >
                &times;
              </button>
            </div>
          {/each}
          {#if recentProjects.length === 0}
            <div class="dropdown-empty">No recent projects</div>
          {/if}
        </div>
      {/if}
    </div>
    <div class="current-project">
      {currentProject ? currentProject.split(/[\\/]/).pop() : 'No Project Loaded'}
    </div>
    <button
      class="toolbar-button"
      onclick={() => onrefresh?.()}
      title="Refresh graph data"
      aria-label="Refresh graph data"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path
          d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"
        />
      </svg>
      Refresh
    </button>
  </div>

  <div class="toolbar-section">
    <div class="view-toggle">
      <button
        class="toggle-button"
        class:active={viewMode === 'batch'}
        onclick={toggleViewMode}
        title="Toggle between batch and cluster view"
        aria-label="Toggle between batch and cluster view"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z" />
        </svg>
        {viewMode === 'batch' ? 'Batch' : 'Cluster'}
      </button>
    </div>
  </div>

  <div class="toolbar-section">
    <button
      class="toolbar-button"
      onclick={() => onimportModel?.()}
      title="Import model"
      aria-label="Import model"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
      </svg>
      Import Model
    </button>

    <button
      class="toolbar-button"
      onclick={() => onaddExternalSource?.()}
      title="Add external audio source"
      aria-label="Add external audio source"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path
          d="M10 4H4c-1.11 0-2 .89-2 2v12c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2h-8l-2-2z"
        />
      </svg>
      Add Source
    </button>

    <div class="file-menu" bind:this={utilitiesMenuElement}>
      <button
        class="toolbar-button"
        onclick={() => (showUtilitiesMenu = !showUtilitiesMenu)}
        aria-label="Open utilities menu"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.06-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.73,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.06,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.43-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.49-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/>
        </svg>
        Utilities
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="currentColor"
          class:rotate-180={showUtilitiesMenu}
          class="chevron"
        >
          <path d="M7 10l5 5 5-5z" />
        </svg>
      </button>

      {#if showUtilitiesMenu}
        <div class="project-dropdown utilities-dropdown">
          <button
            class="dropdown-item"
            onclick={() => {
              onupdateEmbeddings?.()
              showUtilitiesMenu = false
            }}
          >
            Update Embeddings
          </button>
          <button class="dropdown-item" onclick={toggleSprings}>
            {springsVisible ? 'Hide Spring Edges' : 'Show Spring Edges'}
          </button>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .toolbar {
    position: relative;
    z-index: 1010;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    background: var(--color-background-glass-1);
    border-bottom: 1px solid var(--color-border-glass-1);
    backdrop-filter: blur(10px);
  }

  .toolbar-section {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .file-menu {
    position: relative;
  }

  .project-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .project-button:hover {
    background: var(--color-background-glass-hover-1);
  }

  .chevron {
    transition: transform 0.2s;
  }

  .rotate-180 {
    transform: rotate(180deg);
  }

  .project-dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    min-width: 260px;
    background: var(--color-background-glass-4);
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.375rem;
    margin-top: 0.25rem;
    max-height: 300px;
    overflow-y: auto;
    z-index: 1000;
    backdrop-filter: blur(10px);
  }

  .utilities-dropdown {
    right: 0;
    left: auto;
  }

  .dropdown-item-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-right: 0.5rem;
  }
  .dropdown-item-container:hover {
    background: var(--color-border-glass-1);
  }

  .remove-recent {
    background: none;
    border: none;
    color: var(--color-text-overlay-secondary);
    cursor: pointer;
    font-size: 1.2rem;
    padding: 0 0.5rem;
    opacity: 0.5;
    transition: opacity 0.2s;
  }
  .remove-recent:hover {
    opacity: 1;
    background: var(--color-background-glass-hover-1);
  }

  .dropdown-header {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    font-weight: 600;
    border-bottom: 1px solid var(--color-border-glass-1);
    color: var(--color-text-overlay-primary);
  }

  .dropdown-item {
    display: block;
    width: 100%;
    padding: 0.5rem 1rem;
    background: none;
    border: none;
    color: var(--color-overlay-text);
    text-align: left;
    cursor: pointer;
    transition: background-color 0.2s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .recent-project-path {
    display: block;
    font-size: 0.75rem;
    color: var(--color-text-overlay-secondary);
    opacity: 0.8;
  }

  .dropdown-divider {
    height: 1px;
    background-color: var(--color-border-glass-1);
    margin: 0.5rem 0;
  }

  .dropdown-empty {
    padding: 1rem;
    text-align: center;
    color: var(--color-text-overlay-secondary);
    font-style: italic;
  }

  .current-project {
    color: var(--color-overlay-text);
    font-size: 0.875rem;
    font-style: italic;
    opacity: 0.7;
  }

  .view-toggle .toggle-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .toggle-button.active {
    background: var(--color-primary-t-30);
    border-color: var(--color-primary-t-50);
  }

  .toolbar-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .toolbar-button:hover {
    background: var(--color-background-glass-hover-1);
    transform: translateY(-1px);
  }
</style>
