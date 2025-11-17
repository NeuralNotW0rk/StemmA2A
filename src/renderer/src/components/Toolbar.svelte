<!-- src/renderer/src/components/Toolbar.svelte -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte'
  
  export let currentProject: string = ''
  export let viewMode: 'batch' | 'cluster' = 'batch'

  const dispatch = createEventDispatcher()

  let showProjectSelector = false
  let availableProjects: string[] = []
  let showImportDialog = false
  let importPath = ''

  onMount(async () => {
    await loadProjects()
  })

  async function loadProjects() {
    try {
      const response = await window.api.listProjects()
      if (response.project_names) {
        availableProjects = response.project_names
      }
    } catch (error) {
      console.error('Failed to load projects:', error)
    }
  }

  function selectProject(projectName: string) {
    dispatch('projectLoad', projectName)
    showProjectSelector = false
  }

  function toggleViewMode() {
    const newMode = viewMode === 'batch' ? 'cluster' : 'batch'
    dispatch('viewModeChange', newMode)
  }

  async function importModel() {
    if (!importPath.trim()) return
    
    try {
      await window.api.importModel(importPath)
      importPath = ''
      showImportDialog = false
      dispatch('refresh')
    } catch (error) {
      console.error('Failed to import model:', error)
    }
  }

  // File dialog simulation (in real app, you'd use Electron's dialog)
  function selectModelPath() {
    // This would open an Electron file dialog
    importPath = '/path/to/model.ckpt'
  }
</script>

<div class="toolbar">
  <div class="toolbar-section">
    <div class="project-selector">
      <button 
        class="project-button"
        on:click={() => showProjectSelector = !showProjectSelector}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M10 4H4c-1.11 0-2 .89-2 2v12c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2h-8l-2-2z"/>
        </svg>
        {currentProject || 'No Project'}
        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" class="chevron">
          <path d="M7 10l5 5 5-5z"/>
        </svg>
      </button>

      {#if showProjectSelector}
        <div class="project-dropdown">
          <div class="dropdown-header">Select Project</div>
          {#each availableProjects as project}
            <button 
              class="dropdown-item"
              class:active={project === currentProject}
              on:click={() => selectProject(project)}
            >
              {project}
            </button>
          {/each}
          {#if availableProjects.length === 0}
            <div class="dropdown-empty">No projects found</div>
          {/if}
        </div>
      {/if}
    </div>
  </div>

  <div class="toolbar-section">
    <div class="view-toggle">
      <button 
        class="toggle-button"
        class:active={viewMode === 'batch'}
        on:click={toggleViewMode}
        title="Toggle between batch and cluster view"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z"/>
        </svg>
        {viewMode === 'batch' ? 'Batch' : 'Cluster'}
      </button>
    </div>
  </div>

  <div class="toolbar-section">
    <button 
      class="toolbar-button"
      on:click={() => showImportDialog = true}
      title="Import model"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
      </svg>
      Import Model
    </button>

    <button 
      class="toolbar-button"
      on:click={() => dispatch('addExternalSource')}
      title="Add external audio source"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M10 4H4c-1.11 0-2 .89-2 2v12c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2h-8l-2-2z"/>
      </svg>
      Add Source
    </button>
  </div>
</div>

{#if showImportDialog}
  <div class="dialog-overlay" on:click={() => showImportDialog = false}>
    <div class="dialog" on:click|stopPropagation>
      <div class="dialog-header">
        <h3>Import Model</h3>
        <button on:click={() => showImportDialog = false}>×</button>
      </div>
      
      <div class="dialog-content">
        <label>
          Model Path:
          <div class="path-input">
            <input 
              type="text" 
              bind:value={importPath}
              placeholder="/path/to/model.ckpt"
            />
            <button on:click={selectModelPath}>Browse</button>
          </div>
        </label>
      </div>

      <div class="dialog-actions">
        <button on:click={() => showImportDialog = false}>Cancel</button>
        <button 
          class="primary" 
          on:click={importModel}
          disabled={!importPath.trim()}
        >
          Import
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    background: rgba(0, 0, 0, 0.3);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
  }

  .toolbar-section {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .project-selector {
    position: relative;
  }

  .project-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .project-button:hover {
    background: rgba(255, 255, 255, 0.15);
  }

  .chevron {
    transition: transform 0.2s;
  }

  .project-selector:hover .chevron {
    transform: rotate(180deg);
  }

  .project-dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: rgba(0, 0, 0, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 0.375rem;
    margin-top: 0.25rem;
    max-height: 200px;
    overflow-y: auto;
    z-index: 1000;
    backdrop-filter: blur(10px);
  }

  .dropdown-header {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    font-weight: 600;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.7);
  }

  .dropdown-item {
    display: block;
    width: 100%;
    padding: 0.5rem 1rem;
    background: none;
    border: none;
    color: white;
    text-align: left;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .dropdown-item:hover {
    background: rgba(255, 255, 255, 0.1);
  }

  .dropdown-item.active {
    background: rgba(147, 51, 234, 0.3);
  }

  .dropdown-empty {
    padding: 1rem;
    text-align: center;
    color: rgba(255, 255, 255, 0.5);
    font-style: italic;
  }

  .view-toggle .toggle-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .toggle-button.active {
    background: rgba(147, 51, 234, 0.3);
    border-color: rgba(147, 51, 234, 0.5);
  }

  .toolbar-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .toolbar-button:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-1px);
  }

  .dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    backdrop-filter: blur(4px);
  }

  .dialog {
    background: #1e293b;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 0.5rem;
    min-width: 400px;
    max-width: 500px;
  }

  .dialog-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .dialog-header h3 {
    margin: 0;
    color: white;
  }

  .dialog-header button {
    background: none;
    border: none;
    color: white;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .dialog-content {
    padding: 1.5rem;
  }

  .dialog-content label {
    display: block;
    color: white;
    font-weight: 500;
    margin-bottom: 0.5rem;
  }

  .path-input {
    display: flex;
    gap: 0.5rem;
  }

  .path-input input {
    flex: 1;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 0.5rem;
    border-radius: 0.375rem;
  }

  .path-input button {
    background: rgba(147, 51, 234, 0.3);
    border: 1px solid rgba(147, 51, 234, 0.5);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
  }

  .dialog-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    padding: 1rem 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }

  .dialog-actions button {
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .dialog-actions button:not(.primary) {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
  }

  .dialog-actions button.primary {
    background: #9333ea;
    border: 1px solid #9333ea;
    color: white;
  }

  .dialog-actions button.primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>