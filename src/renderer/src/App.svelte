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
  import BatchingView from './components/views/BatchingView.svelte'
  import JobStatusView from './components/views/JobStatusView.svelte'
  import { onMount, onDestroy } from 'svelte'
  import {
    initiatorNodeStore,
    contextStore,
    selectedForRemoval,
    backendStatus,
    isCreatingNewProject
  } from './utils/stores'
  import { startEmbeddingUpdate } from './utils/execution'
  import { jobStore, addJob, updateJob } from './utils/job-management'
  import type { ActionPanelView, ElementData, ErrorInfo } from './utils/types'

  let graphData: any = $state(null)
  let currentProject: string | null = $state(null)
  let viewMode: 'batch' | 'cluster' = $state('batch')
  let audioSrc: string | null = $state(null)
  let audioTitle: string | null = $state(null)
  let showSpringEdges = $state(false)
  let selectedElementData: ElementData | null = $state(null)
  let actionPanelView: ActionPanelView = $state('none')
  let errorInInfoPanel: ErrorInfo | null = $state(null)
  let toolbarComponent: Toolbar

  let isWaitingForBackend = $state(true)
  // Backend restart detection
  let serverInstanceId: string | null = $state(null)
  let healthCheckInterval: number | null = null

  $effect(() => {
    const unsub = jobStore.subscribe((jobs) => {
      // Find jobs that have just succeeded
      const newlySucceeded = jobs.find((job) => job.status === 'success' && !job.result?.viewed)

      if (newlySucceeded) {
        refreshGraphData()
        // Mark as viewed to prevent re-triggering
        if (newlySucceeded.result) {
          newlySucceeded.result.viewed = true
        }
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
      isWaitingForBackend = false
    } catch (error) {
      console.error('Failed to get backend health:', error)
      const MAX_RETRIES = 20
      const RETRY_INTERVAL = 3000
      for (let i = 0; i < MAX_RETRIES; i++) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_INTERVAL))
        try {
          const status = await window.api.getHealth()
          backendStatus.set(status)
          serverInstanceId = status.server_instance_id || null
          isWaitingForBackend = false
          return
        } catch (e) {
          console.error(`Backend connection attempt ${i + 1} failed:`, e)
        }
      }
      isWaitingForBackend = false
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
    try {
      await startEmbeddingUpdate()
    } catch (error: any) {
      console.error('Failed to update embeddings:', error)
      errorInInfoPanel = {
        title: 'Update Embeddings Failed',
        message: error?.message || String(error)
      }
    }
  }

  async function handleUpdateLabels(): Promise<void> {
    try {
      await window.api.updateLabels()
      await refreshGraphData()
    } catch (error: any) {
      console.error('Failed to update labels:', error)
      errorInInfoPanel = {
        title: 'Update Labels Failed',
        message: error?.message || String(error)
      }
    }
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

  async function handleAddExternalSource(path: string): Promise<void> {
    try {
      await window.api.addExternalSource(path)
      await refreshGraphData()
    } catch (error: any) {
      console.error('Error adding external source:', error)
      errorInInfoPanel = { title: 'Add Source Failed', message: error.message || String(error) }
    }
  }

  async function handleExpandPath(pathNodeId: string): Promise<void> {
    // 1. Register the job locally before the backend blocks
    const job = addJob('Expand Directory', { pathNodeId }, 'running')

    try {
      const response = await window.api.expandPath(pathNodeId)

      // 2. When the backend finally unblocks, mark it successful
      job.status = 'success'
      job.result = { id: response.directory_id, viewed: false }
      updateJob(job)

      await refreshGraphData()
    } catch (error: any) {
      console.error('Error expanding path:', error)
      errorInInfoPanel = { title: 'Expand Path Failed', message: error.message || String(error) }

      // 3. Capture any failures
      job.status = 'error'
      job.error = { title: 'Expansion Failed', message: error.message || String(error) }
      updateJob(job)
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
    contextStore.set({ model_id: modelData.id })
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

  function handleElementRemove(elementData: any): void {
    selectedForRemoval.set(elementData)
    actionPanelView = 'removal'
  }

  function handleStartBatching(nodeData: any): void {
    initiatorNodeStore.set(nodeData)
    actionPanelView = 'batching'
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
    if (actionPanelView === 'batching') {
      return ($initiatorNodeStore?.type as string) === 'batch' ? 'Update Batch' : 'Create Batch'
    }
    return 'Action'
  }

  async function handleSavePositions(
    positions: Record<string, { x: number; y: number }>
  ): Promise<void> {
    if (!currentProject) return
    try {
      // You will need to implement this API endpoint in your preload/main scripts
      await window.api.saveNodePositions(currentProject, positions)
    } catch (error) {
      console.error('Failed to save node positions:', error)
    }
  }
</script>

<main class="container">
  {#if isWaitingForBackend}
    <div class="backend-wait-overlay">
      <div class="spinner" style="width: 32px; height: 32px; border-width: 4px;"></div>
      <p>Waiting for backend server...</p>
    </div>
  {/if}

  <Toolbar
    bind:this={toolbarComponent}
    onprojectLoad={handleProjectLoad}
    onviewModeChange={handleViewModeChange}
    onrefresh={refreshGraphData}
    onimportModel={() => (actionPanelView = 'import-model')}
    onaddExternalSource={handleAddExternalSource}
    onupdateEmbeddings={handleUpdateEmbeddings}
    onupdateLabels={handleUpdateLabels}
    {currentProject}
    {viewMode}
    bind:showSpringEdges
  />

  {#if errorInInfoPanel}
    <ContentPanel
      title={errorInInfoPanel.title}
      onclose={() => (errorInInfoPanel = null)}
      position="left"
    >
      <ErrorView title={errorInInfoPanel.title} message={errorInInfoPanel.message} />
    </ContentPanel>
  {:else if actionPanelView !== 'none'}
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
      {:else if actionPanelView === 'batching'}
        <BatchingView
          onclose={closeActionPanel}
          onrefresh={() => {
            closeActionPanel()
            refreshGraphData()
          }}
          onerror={(error) => {
            console.error('Batching error:', error)
            closeActionPanel()
            errorInInfoPanel = error
          }}
        />
      {/if}
    </ContentPanel>
  {:else if $isCreatingNewProject}
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

  <div class="right-panel-container">
    {#if $jobStore.length > 0}
      <ContentPanel title="Jobs" position="right">
        <JobStatusView />
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
  </div>

  <ParameterGraph
    {graphData}
    {showSpringEdges}
    {viewMode}
    onaudioSelect={handleAudioSelect}
    onmodelSelect={handleModelSelect}
    onaudioNodeSelectForGeneration={handleAudioNodeSelectForGeneration}
    onnodeSelect={handleElementSelect}
    onedgeSelect={handleElementSelect}
    onElementRemove={handleElementRemove}
    onstartBatching={handleStartBatching}
    onsavePositions={handleSavePositions}
    onexpandPath={handleExpandPath}
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

  .right-panel-container {
    position: absolute;
    top: 4.5rem;
    right: 1rem;
    bottom: 1rem;
    width: 320px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    z-index: 1000;
    pointer-events: none;
  }

  .right-panel-container > :global(*) {
    pointer-events: auto;
    flex-shrink: 1;
    min-height: 12rem;
    position: relative !important;
    top: auto !important;
    bottom: auto !important;
    left: auto !important;
    right: auto !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden;
  }

  .right-panel-container > :global(*) > :global(*:not(:first-child)) {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    min-height: 0 !important;
    flex-grow: 1 !important;
  }

  :global(body) {
    overflow: hidden;
    background-image: url('./assets/wavy-lines.svg');
    background-size: cover;
    user-select: none;
  }

  :global(*) {
    scrollbar-width: thin;
    scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
  }

  :global(::-webkit-scrollbar) {
    width: 8px;
    height: 8px;
  }

  :global(::-webkit-scrollbar-track) {
    background: transparent;
  }

  :global(::-webkit-scrollbar-thumb) {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
  }

  :global(::-webkit-scrollbar-thumb:hover) {
    background: rgba(255, 255, 255, 0.3);
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

  .backend-wait-overlay {
    position: absolute;
    inset: 0;
    background-color: rgba(15, 15, 20, 0.7);
    backdrop-filter: blur(8px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    color: var(--color-overlay-text, #fff);
    font-size: 1.1rem;
    gap: 1.5rem;
  }
</style>
