<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  import ParameterGraph from './components/graph/ParameterGraph.svelte'
  import Toolbar from './components/Toolbar.svelte'
  import AudioPlayer from './components/AudioPlayer.svelte'
  import ContentPanel from './components/ContentPanel.svelte'
  import ElementInfoView from './components/views/ElementInfoView.svelte'
  import GenerationView from './components/views/GenerationView.svelte'
  import ErrorView from './components/views/ErrorView.svelte'
  import ImportModelView from './components/views/ImportModelView.svelte'
  import RemovalView from './components/views/RemovalView.svelte'
  import NewProjectView from './components/views/NewProjectView.svelte'
  import ExecutionView from './components/views/ExecutionView.svelte'
  import { onMount, onDestroy } from 'svelte'
  import {
    initiatorNodeStore,
    contextStore,
    selectedForRemoval,
    backendStatus,
    isCreatingNewProject
  } from './utils/stores'
  import { executionStore, embeddingUpdateExecutionStore, startEmbeddingUpdate } from './utils/execution'

  type ActionPanelView = 'generation' | 'import-model' | 'removal' | 'none'

  let graphData: any = $state(null)
  let currentProject: string | null = $state(null)
  let viewMode: 'batch' | 'cluster' = $state('batch')
  let audioSrc: string | null = $state(null)
  let audioTitle: string | null = $state(null)
  let selectedElementData: Record<string, any> | null = $state(null)
  let actionPanelView: ActionPanelView = $state('none')
  let errorInInfoPanel: { title: string; message: string } | null = $state(null)
  let toolbarComponent: Toolbar

  // Backend restart detection
  let serverInstanceId: string | null = $state(null)
  let healthCheckInterval: number | null = null

  $effect(() => {
    const unsub = executionStore.subscribe((value) => {
      if (value.status === 'success') {
        refreshGraphData()
      }
    })
    return unsub
  })

  $effect(() => {
    const unsub = embeddingUpdateExecutionStore.subscribe((value) => {
      if (value.status === 'success') {
        refreshGraphData()
      }
    })
    return unsub
  })

  onMount(async () => {
    await initialHealthCheck()
    setupHealthCheckPolling()
  })

  onDestroy(() => {
    if (healthCheckInterval) {
      clearInterval(healthCheckInterval)
    }
  })

  async function initialHealthCheck(): Promise<void> {
    try {
      const status = await window.api.getHealth()
      backendStatus.set(status)
      serverInstanceId = status.server_instance_id || null
      console.log('Backend status:', status)
    } catch (error) {
      console.error('Failed to get backend health:', error)
      errorInInfoPanel = {
        title: 'Backend Connection Failed',
        message: 'Waiting for backend...'
      }
      const MAX_RETRIES = 10
      const RETRY_INTERVAL = 3000
      for (let i = 0; i < MAX_RETRIES; i++) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_INTERVAL))
        try {
          const status = await window.api.getHealth()
          backendStatus.set(status)
          serverInstanceId = status.server_instance_id || null
          errorInInfoPanel = null
          return
        } catch (e) {
          console.error(`Backend connection attempt ${i + 1} failed:`, e)
          errorInInfoPanel = {
            title: 'Backend Connection Failed',
            message: `Waiting for backend... (attempt ${i + 2}/${MAX_RETRIES})`
          }
        }
      }
      errorInInfoPanel = {
        title: 'Backend Connection Failed',
        message:
          'Could not connect to the backend server. Please ensure it is running and accessible.'
      }
    }
  }

  function setupHealthCheckPolling(): void {
    if (healthCheckInterval) {
      clearInterval(healthCheckInterval)
    }
    healthCheckInterval = window.setInterval(async () => {
      try {
        const status = await window.api.getHealth()
        backendStatus.set(status)

        if (serverInstanceId && status.server_instance_id !== serverInstanceId) {
          console.warn('Backend has restarted. Resetting project state.')
          handleBackendRestart()
        }
        serverInstanceId = status.server_instance_id || null
      } catch (error) {
        console.error('Health check poll failed:', error)
        // Optionally handle polling errors, e.g., show a banner
      }
    }, 3000) // Poll every 3 seconds
  }

  function handleBackendRestart(): void {
    currentProject = null
    graphData = null
    selectedElementData = null
    audioSrc = null
    audioTitle = null
    actionPanelView = 'none'
    $isCreatingNewProject = false

    // Close any selection process
    initiatorNodeStore.set(null)
    selectedForRemoval.set(null)

    errorInInfoPanel = {
      title: 'Backend Restarted',
      message: 'The backend server has restarted. Please reload your project.'
    }
  }

  async function handleUpdateEmbeddings(): Promise<void> {
    await startEmbeddingUpdate()
  }

  function closeActionPanel(): void {
    actionPanelView = 'none'
    initiatorNodeStore.set(null)
    contextStore.set(null)
    selectedForRemoval.set(null)
  }

  async function handleProjectLoad(data: {
    project_path?: string
    project_name?: string
  }): Promise<void> {
    try {
      const projectName = data.project_name || data.project_path?.split(/[/\\]/).pop()
      console.log(`Loading project: ${projectName}`)
      await window.api.loadProject(data)
      graphData = await window.api.getGraphData(viewMode)
      currentProject = projectName || null
      if (data.project_path) {
        await window.api.addRecentProject(data.project_path)
      }
      console.log(`Successfully loaded project: ${projectName}`)
    } catch (error) {
      console.error('Failed to load project:', error)
      errorInInfoPanel = { title: 'Project Load Failed', message: error.message }
    }
  }

  async function handleProjectCreate(data: {
    project_path?: string
    project_name?: string
  }): Promise<void> {
    const projectName = data.project_name || data.project_path?.split(/[/\\]/).pop()
    console.log(`Creating project: ${projectName} at ${data.project_path}`)
    await window.api.createProject(data) // Pass the whole data object
    graphData = await window.api.getGraphData(viewMode)
    currentProject = projectName || null
    if (data.project_path) {
      await window.api.addRecentProject(data.project_path)
    }
    console.log(`Successfully created project: ${projectName}`)
  }

  async function handleProjectCreateAndRefresh(data: {
    project_path: string
    project_name: string
  }): Promise<void> {
    try {
      await handleProjectCreate(data)
      await toolbarComponent.refreshRecentProjects()
    } catch (error: any) {
      console.error('Failed to create project:', error)
      // Re-throw the error so the child component can display it
      throw new Error(error.message || 'An unknown error occurred during project creation.')
    }
  }

  async function handleViewModeChange(mode: 'batch' | 'cluster'): Promise<void> {
    viewMode = mode
    try {
      graphData = await window.api.getGraphData(viewMode)
    } catch (error) {
      console.error('Failed to get graph data:', error)
      errorInInfoPanel = { title: 'View Change Failed', message: error.message }
    }
  }

  async function handleAudioSelect(audioData: any): Promise<void> {
    console.log(`Audio selected: ${audioData.name}`)

    if (audioSrc && audioSrc.startsWith('blob:')) {
      URL.revokeObjectURL(audioSrc)
    }

    try {
      const result = await window.api.getAudioFile(audioData.id)
      if (result && result.buffer) {
        const { buffer, mimeType } = result
        const blob = new Blob([new Uint8Array(buffer)], { type: mimeType })
        audioSrc = URL.createObjectURL(blob)
        audioTitle = audioData.name
      } else {
        throw new Error('Failed to get audio file buffer.')
      }
    } catch (error) {
      console.error('Error getting audio file:', error)
      errorInInfoPanel = { title: 'Audio Load Failed', message: error.message }
      audioSrc = null
      audioTitle = null
    }
  }

  function handleElementSelect(elementData: any): void {
    selectedElementData = elementData
    errorInInfoPanel = null // Clear any existing errors when a new element is selected
    console.log(`Element selected: ${selectedElementData?.name || 'Unknown'}`)
  }

  function handleModelSelect(modelData: any): void {
    initiatorNodeStore.set(modelData)
    contextStore.set(null)
    actionPanelView = 'generation'
  }

  function handleAudioNodeSelectForGeneration(audioData: any, useContext?: boolean): void {
    initiatorNodeStore.set(audioData)
    if (useContext) {
      contextStore.set(audioData.context)
    } else {
      contextStore.set(null)
    }
    actionPanelView = 'generation'
  }

  function handleNodeRemove(nodeData: any): void {
    selectedForRemoval.set(nodeData)
    actionPanelView = 'removal'
  }

  function handleGenerationError(error: { title: string; message: string }): void {
    errorInInfoPanel = error
  }

  async function refreshGraphData(): Promise<void> {
    try {
      graphData = await window.api.getGraphData(viewMode)
      console.log(`Graph data refreshed for view mode ${viewMode}`)
    } catch (error) {
      console.error('Failed to get graph data:', error)
      errorInInfoPanel = { title: 'View Change Failed', message: error.message }
    }
  }

  function getActionPanelTitle(): string {
    if (actionPanelView === 'import-model') {
      return 'Import Model'
    }
    if (actionPanelView === 'generation') {
      return 'Generation'
    }
    if (actionPanelView === 'removal') {
      return 'Confirm Removal'
    }
    return 'Action'
  }
</script>

<main class="container">
  <Toolbar
    bind:this={toolbarComponent}
    onprojectLoad={handleProjectLoad}
    onviewModeChange={handleViewModeChange}
    onrefresh={refreshGraphData}
    onimportModel={() => (actionPanelView = 'import-model')}
    onupdateEmbeddings={handleUpdateEmbeddings}
    {currentProject}
    {viewMode}
  />

  {#if $embeddingUpdateExecutionStore.status !== 'idle'}
    <ContentPanel
      title="Updating Embeddings"
      onclose={() => {
        /* This panel cannot be closed directly */
      }}
      position="left"
    >
      <ExecutionView executionStore={embeddingUpdateExecutionStore} />
    </ContentPanel>
  {:else if $executionStore.status !== 'idle'}
    <ContentPanel
      title="Generation"
      onclose={() => {
        /* This panel cannot be closed directly */
      }}
      position="left"
    >
      <ExecutionView />
    </ContentPanel>
  {:else if errorInInfoPanel}
    <ContentPanel
      title={errorInInfoPanel.title}
      onclose={() => (errorInInfoPanel = null)}
      position="left"
    >
      <ErrorView title={errorInInfoPanel.title} message={errorInInfoPanel.message} />
    </ContentPanel>
  {/if}

  {#if selectedElementData}
    <ContentPanel
      title={selectedElementData.alias || selectedElementData.name || 'Element Details'}
      onclose={() => {
        selectedElementData = null
      }}
      position="right"
    >
      <ElementInfoView {selectedElementData} />
    </ContentPanel>
  {/if}

  {#if actionPanelView !== 'none' && $executionStore.status === 'idle' && !errorInInfoPanel}
    <ContentPanel title={getActionPanelTitle()} onclose={closeActionPanel} position="left">
      {#if actionPanelView === 'generation'}
        <GenerationView
          onClose={closeActionPanel}
          onGenerate={closeActionPanel}
          onError={handleGenerationError}
        />
      {:else if actionPanelView === 'import-model'}
        <ImportModelView
          onclose={closeActionPanel}
          onrefresh={() => {
            closeActionPanel()
            refreshGraphData()
          }}
          onError={(error) => {
            console.error('Import error:', error)
            closeActionPanel()
            errorInInfoPanel = error
          }}
        />
      {:else if actionPanelView === 'removal'}
        <RemovalView
          onclose={closeActionPanel}
          onrefresh={() => {
            closeActionPanel()
            refreshGraphData()
          }}
          onerror={(error) => {
            console.error('Removal error:', error)
            closeActionPanel()
            errorInInfoPanel = error
          }}
        />
      {/if}
    </ContentPanel>
  {/if}

  {#if $isCreatingNewProject}
    <ContentPanel
      title="Create New Project"
      onclose={() => ($isCreatingNewProject = false)}
      position="left"
    >
      <NewProjectView
        onclose={() => ($isCreatingNewProject = false)}
        oncreate={handleProjectCreateAndRefresh}
      />
    </ContentPanel>
  {/if}

  <ParameterGraph
    {graphData}
    {viewMode}
    onaudioSelect={handleAudioSelect}
    onmodelSelect={handleModelSelect}
    onaudioNodeSelectForGeneration={handleAudioNodeSelectForGeneration}
    onnodeSelect={handleElementSelect}
    onedgeSelect={handleElementSelect}
    onnodeRemove={handleNodeRemove}
  />
  {#if audioSrc}
    <AudioPlayer src={audioSrc} title={audioTitle} onclose={() => (audioSrc = null)} />
  {/if}
</main>

<style>
  .container {
    position: relative;
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  :global(body) {
    overflow: hidden;
    background-image: url('./assets/wavy-lines.svg');
    background-size: cover;
    user-select: none;
  }

  :global(button) {
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
    min-width: 80px;
    min-height: 38px;
    border: 1px solid var(--color-overlay-border-primary);
    background: var(--color-border-glass-1);
    color: var(--color-overlay-text);
  }

  :global(button.primary) {
    background: var(--color-primary);
    border: 1px solid var(--color-primary);
    color: var(--color-overlay-text);
    display: flex;
    justify-content: center;
    align-items: center;
  }

  :global(button:disabled) {
    opacity: 0.5;
    cursor: not-allowed;
  }

  :global(.spinner) {
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-left-color: #fff;
    border-radius: 50%;
    width: 18px;
    height: 18px;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
