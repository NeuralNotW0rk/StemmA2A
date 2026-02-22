<script lang="ts">
  import { onMount } from 'svelte'

  interface Props {
    currentProject?: string | null
    viewMode?: 'batch' | 'cluster'
    onprojectLoad?: (data: { projectPath: string }) => void
    onprojectCreate?: (data: { projectPath: string }) => void
    onviewModeChange?: (mode: 'batch' | 'cluster') => void
    onrefresh?: () => void
    onaddExternalSource?: () => void
    onimportModel?: () => void
  }

  let {
    currentProject = null,
    viewMode = 'batch',
    onprojectLoad,
    onprojectCreate,
    onviewModeChange,
    onrefresh,
    onaddExternalSource,
    onimportModel
  }: Props = $props()

  let showFileMenu = $state(false)
  let fileMenuElement: HTMLElement | undefined = $state()
  let recentProjects: string[] = $state([])

  onMount(async (): Promise<void> => {
    recentProjects = await window.api.getRecentProjects()
  })

  async function openProject(): Promise<void> {
    showFileMenu = false
    const path = await window.api.openProject()
    if (path) {
      onprojectLoad?.({ projectPath: path })
    }
  }

  async function newProject(): Promise<void> {
    showFileMenu = false
    const path = await window.api.newProject()
    if (path) {
      onprojectCreate?.({ projectPath: path })
    }
  }

  function loadRecentProject(path: string): void {
    showFileMenu = false
    onprojectLoad?.({ projectPath: path })
  }

  async function removeRecentProject(path: string): Promise<void> {
    await window.api.removeRecentProject(path)
    recentProjects = recentProjects.filter((p) => p !== path)
  }

  function toggleViewMode(): void {
    const newMode = viewMode === 'batch' ? 'cluster' : 'batch'
    onviewModeChange?.(newMode)
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape' && showFileMenu) {
      showFileMenu = false
    }
  }
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
          <button class="dropdown-item" onclick={newProject}> New Project... </button>
          <button class="dropdown-item" onclick={openProject}> Load Project... </button>
          <div class="dropdown-divider"></div>
          <div class="dropdown-header">Recent Projects</div>
          {#each recentProjects as project (project)}
            <div class="dropdown-item-container">
              <button class="dropdown-item" onclick={() => loadRecentProject(project)}>
                {project.split(/[\\/]/).pop()}
              </button>
              <button
                class="remove-project-button"
                onclick={(e) => {
                  e.stopPropagation()
                  removeRecentProject(project)
                }}
                title="Remove from recent projects"
                aria-label="Remove from recent projects"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                  <path
                    d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"
                  />
                </svg>
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
    min-width: 200px;
    background: var(--color-background-glass-4);
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.375rem;
    margin-top: 0.25rem;
    max-height: 200px;
    overflow-y: auto;
    z-index: 1000;
    backdrop-filter: blur(10px);
  }

  .dropdown-item-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .dropdown-item-container:hover .remove-project-button {
    opacity: 1;
  }

  .remove-project-button {
    background: transparent;
    border: none;
    color: var(--color-overlay-text);
    cursor: pointer;
    padding: 0.5rem;
    opacity: 0.5;
    transition: opacity 0.2s;
  }

  .remove-project-button:hover {
    color: var(--color-primary);
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
  }

  .dropdown-item:hover {
    background: var(--color-border-glass-1);
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
