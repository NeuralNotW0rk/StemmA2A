<script lang="ts">
  import { onMount } from 'svelte'
  import { isCreatingNewProject } from '../utils/stores'
  import EngineMenu from './EngineMenu.svelte'
  import type { EngineSettings, EngineState, ProjectOrigin } from '../utils/types'
  import type { RoomCrumb } from '../utils/rooms'

  interface Props {
    engine?: EngineState | null
    onswitchEngine?: (settings: EngineSettings) => Promise<void>
    currentProject?: string | null
    projectOrigin?: ProjectOrigin | null
    onreturnToOrigin?: () => void
    // Rooms from the outermost to the one being viewed; empty in the main graph
    roomCrumbs?: RoomCrumb[]
    onnavigateRoom?: (roomId: string | null) => void
    onprojectLoad?: (data: { project_path?: string; project_name?: string }) => void
    onrefresh?: () => void
    onaddExternalSource?: (path: string) => void
    onimportModel?: () => void
    onupdateEmbeddings?: () => void
    onupdateLabels?: () => void
    onrepairEdges?: () => void
    oncleanup?: () => void
    showSpringEdges?: boolean
    showDetailedLabels?: boolean
    showEdgeLabels?: boolean
    chronologicalConstraint?: boolean
    hideEvolutionaryEdges?: boolean
    hideAllEdges?: boolean
  }

  let {
    engine = null,
    onswitchEngine,
    currentProject = null,
    projectOrigin = null,
    onreturnToOrigin,
    roomCrumbs = [],
    onnavigateRoom,
    onprojectLoad,
    onrefresh,
    onaddExternalSource,
    onimportModel,
    onupdateEmbeddings,
    onupdateLabels,
    onrepairEdges,
    oncleanup,
    showSpringEdges = $bindable(false),
    showDetailedLabels = $bindable(false),
    showEdgeLabels = $bindable(false),
    chronologicalConstraint = $bindable(false),
    hideEvolutionaryEdges = $bindable(false),
    hideAllEdges = $bindable(false)
  }: Props = $props()

  let showFileMenu = $state(false)
  let fileMenuElement: HTMLElement | undefined = $state()
  let recentProjects: string[] = $state([])
  let showUtilitiesMenu = $state(false)
  let utilitiesMenuElement: HTMLElement | undefined = $state()

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

  // Shown in the tip view while each control is hovered or focused
  const TIPS = {
    file: 'Start a new project, open one, or reopen a recent one. A project is a folder holding its graph and every file generated in it.',
    newProject:
      'Creates an empty project folder and opens it. Models, sources and everything you generate are added to it as you work.',
    openProject: 'Opens an existing project folder, picking up its graph where you left it.',
    removeRecent:
      'Removes this project from the Recent Projects list. The project folder itself is not touched.',
    breadcrumb:
      "The open project, followed by the room you're in, if any. Click an earlier name to go back up to it.",
    forkedBreadcrumb:
      'This project is a fork: a selection copied out of another project, with everything it depends on. The button to the left returns to that original project.',
    refresh:
      'Reloads the graph from the backend. The graph already updates itself when jobs finish, so this is only needed if the view looks out of date, for example after files changed outside the app.',
    importModel:
      "Adds a generative model to the project so you can generate with it: a Stable Audio or StyleGAN2 checkpoint on disk, or a model the engine has shared. Files aren't copied; the model is identified by a hash of its contents, so large checkpoints take a while.",
    addSource:
      'Links a folder of your own audio or images into the project as a folder node. Right-click it and choose Expand to load its files, which you can then use as initial audio, invert, or wrap as an individual. The files stay where they are; nothing is copied.',
    utilities: 'Maintenance tasks for the project, and options for what the graph shows.',
    updateEmbeddings:
      'Recomputes the similarity embeddings that place related sounds and images near each other, and the spring edges between them. New items are embedded automatically, so this is mainly for refreshing a whole project. Runs as a job.',
    updateLabels:
      'Re-derives the automatic labels of groups and their members from what they contain.',
    repairEdges:
      'Rebuilds the structural edges, such as generated from and parent of, from what each node recorded about how it was made. Use it if edges look missing or wrong, for example after an interrupted job.',
    cleanup:
      'Finds files in the project folder that no element uses any more, and moves the ones you choose to the project trash, where they can be restored or deleted for good. Unavailable while jobs are running.',
    evolutionaryEdges:
      'Hides or shows the lineage edges between individuals. While hidden, they still appear for the node you select, so a large lineage stays readable.',
    allEdges:
      'Hides or shows every edge except the spring (similarity) edges. While hidden, edges still appear for the node you select.',
    springEdges:
      'Shows or hides the similarity edges, the springs that pull related items together in the layout.',
    detailedLabels:
      "Shows each node's alias or prompt along with its name. Hold Alt to peek at detailed labels without turning them on.",
    edgeLabels: 'Labels each edge with what it means, such as generated, binds to or parent of.',
    chronological:
      'Lays generated items out from left to right in the order they were made, so a session reads like a timeline.'
  } as const

  async function handleAddExternalSource(): Promise<void> {
    try {
      const sourcePath = await window.api.selectDirectory()
      if (sourcePath) {
        onaddExternalSource?.(sourcePath)
      }
    } catch (error) {
      console.error('Failed to select external source directory:', error)
    }
  }

  async function handleRemoveRecent(path: string, event: MouseEvent): Promise<void> {
    event.stopPropagation() // Prevent the dropdown item from being clicked
    await window.api.removeRecentProject(path)
    await refreshRecentProjects()
  }

  function toggleEvolutionaryEdges(): void {
    hideEvolutionaryEdges = !hideEvolutionaryEdges
    showUtilitiesMenu = false
  }

  function toggleAllEdges(): void {
    hideAllEdges = !hideAllEdges
    showUtilitiesMenu = false
  }

  function toggleSprings(): void {
    showSpringEdges = !showSpringEdges
    showUtilitiesMenu = false
  }

  function toggleDetailedLabels(): void {
    showDetailedLabels = !showDetailedLabels
    showUtilitiesMenu = false
  }

  function toggleEdgeLabels(): void {
    showEdgeLabels = !showEdgeLabels
    showUtilitiesMenu = false
  }

  function toggleChronological(): void {
    chronologicalConstraint = !chronologicalConstraint
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
        data-tip={TIPS.file}
        data-tip-title="File"
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
            data-tip={TIPS.newProject}
            data-tip-title="New Project"
          >
            New Project...
          </button>
          <button
            class="dropdown-item"
            onclick={openProject}
            data-tip={TIPS.openProject}
            data-tip-title="Open Project"
          >
            Open Project...
          </button>
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
                data-tip={TIPS.removeRecent}
                data-tip-title="Remove from Recent"
                aria-label="Remove from recent"
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
    {#if projectOrigin}
      <button
        class="toolbar-button"
        onclick={() => onreturnToOrigin?.()}
        data-tip={`Return to ${projectOrigin.project_name}, the project this was forked from`}
        data-tip-title="Return to Origin"
        aria-label={`Return to ${projectOrigin.project_name}`}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" />
        </svg>
        {projectOrigin.project_name}
      </button>
    {/if}
    <div
      class="current-project"
      data-tip={projectOrigin ? TIPS.forkedBreadcrumb : TIPS.breadcrumb}
      data-tip-title={projectOrigin ? 'Forked Project' : 'Project'}
    >
      {#if roomCrumbs.length > 0}
        <button class="crumb" onclick={() => onnavigateRoom?.(null)}>
          {currentProject ? currentProject.split(/[\\/]/).pop() : 'Main'}
        </button>
        {#each roomCrumbs as crumb, i (crumb.id)}
          <span class="crumb-separator">›</span>
          {#if i < roomCrumbs.length - 1}
            <button class="crumb" onclick={() => onnavigateRoom?.(crumb.id)}>{crumb.name}</button>
          {:else}
            <span class="crumb current">{crumb.name}</span>
          {/if}
        {/each}
      {:else}
        {currentProject ? currentProject.split(/[\\/]/).pop() : 'No Project Loaded'}
      {/if}
    </div>
    <button
      class="toolbar-button"
      onclick={() => onrefresh?.()}
      data-tip={TIPS.refresh}
      data-tip-title="Refresh"
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
    <EngineMenu {engine} onswitch={onswitchEngine} />

    <button
      class="toolbar-button"
      onclick={() => onimportModel?.()}
      data-tip={TIPS.importModel}
      data-tip-title="Import Model"
      aria-label="Import model"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
      </svg>
      Import Model
    </button>

    <button
      class="toolbar-button"
      onclick={handleAddExternalSource}
      data-tip={TIPS.addSource}
      data-tip-title="Add Source"
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
        data-tip={TIPS.utilities}
        data-tip-title="Utilities"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.06-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.73,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.06,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.43-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.49-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"
          />
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
            data-tip={TIPS.updateEmbeddings}
            data-tip-title="Update Embeddings"
          >
            Update Embeddings
          </button>
          <button
            class="dropdown-item"
            onclick={() => {
              onupdateLabels?.()
              showUtilitiesMenu = false
            }}
            data-tip={TIPS.updateLabels}
            data-tip-title="Update Labels"
          >
            Update Labels
          </button>
          <button
            class="dropdown-item"
            onclick={() => {
              onrepairEdges?.()
              showUtilitiesMenu = false
            }}
            data-tip={TIPS.repairEdges}
            data-tip-title="Repair Edges"
          >
            Repair Edges
          </button>
          <button
            class="dropdown-item"
            onclick={() => {
              oncleanup?.()
              showUtilitiesMenu = false
            }}
            data-tip={TIPS.cleanup}
            data-tip-title="Clean Up Project Files"
          >
            Clean Up Project Files...
          </button>
          <div class="dropdown-divider"></div>
          <button
            class="dropdown-item"
            onclick={toggleEvolutionaryEdges}
            data-tip={TIPS.evolutionaryEdges}
            data-tip-title="Evolutionary Edges"
          >
            {hideEvolutionaryEdges
              ? 'Show Evolutionary Edges'
              : 'Hide Evolutionary Edges (Show on Select)'}
          </button>
          <button
            class="dropdown-item"
            onclick={toggleAllEdges}
            data-tip={TIPS.allEdges}
            data-tip-title="All Edges"
          >
            {hideAllEdges ? 'Show All Edges' : 'Hide All Edges (Show on Select)'}
          </button>
          <button
            class="dropdown-item"
            onclick={toggleSprings}
            data-tip={TIPS.springEdges}
            data-tip-title="Spring Edges"
          >
            {showSpringEdges ? 'Hide Spring Edges' : 'Show Spring Edges'}
          </button>
          <button
            class="dropdown-item"
            onclick={toggleDetailedLabels}
            data-tip={TIPS.detailedLabels}
            data-tip-title="Detailed Labels"
          >
            {showDetailedLabels ? 'Hide Detailed Labels' : 'Show Detailed Labels'}
          </button>
          <button
            class="dropdown-item"
            onclick={toggleEdgeLabels}
            data-tip={TIPS.edgeLabels}
            data-tip-title="Edge Labels"
          >
            {showEdgeLabels ? 'Hide Edge Labels' : 'Show Edge Labels'}
          </button>
          <button
            class="dropdown-item"
            onclick={toggleChronological}
            data-tip={TIPS.chronological}
            data-tip-title="Chronological Layout"
          >
            {chronologicalConstraint
              ? 'Disable Chronological Layout'
              : 'Enable Chronological Layout'}
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

  .crumb {
    background: none;
    border: none;
    padding: 0;
    color: inherit;
    font: inherit;
  }

  button.crumb {
    cursor: pointer;
    text-decoration: underline;
  }

  .crumb.current {
    font-weight: bold;
    font-style: normal;
  }

  .crumb-separator {
    margin: 0 0.35rem;
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
