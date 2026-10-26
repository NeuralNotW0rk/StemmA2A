<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  import ParameterGraph from './components/graph/ParameterGraph.svelte'
  import Toolbar from './components/Toolbar.svelte'
  import AudioPlayer from './components/AudioPlayer.svelte'
  import ContentPanel from './components/ContentPanel.svelte'
  import ElementInfoView from './components/views/ElementInfoView.svelte'
  import MessageView from './components/views/MessageView.svelte'
  import ImportGratingView from './components/views/ImportGratingView.svelte'
  import ImportModelView from './components/views/ImportModelView.svelte'
  import RemovalView from './components/views/RemovalView.svelte'
  import NewProjectView from './components/views/NewProjectView.svelte'
  import GroupingView from './components/views/GroupingView.svelte'
  import JobStatusView from './components/views/JobStatusView.svelte'
  import OperationView from './components/views/OperationView.svelte'
  import BendingView from './components/views/BendingView.svelte'
  import { onMount, onDestroy } from 'svelte'
  import { SvelteSet } from 'svelte/reactivity'
  import {
    initiatorNodeStore,
    contextStore,
    selectedForRemoval,
    backendStatus,
    isCreatingNewProject,
    selectionStore,
    selectedOperation
  } from './utils/stores'
  import { startEmbeddingUpdate } from './utils/execution'
  import {
    jobStore,
    addJob,
    updateJob,
    pollJobStatus,
    clearDismissableJobs,
    cancelAllJobs
  } from './utils/job-management'
  import type {
    ActionPanelView,
    EngineSettings,
    EngineState,
    ForkResult,
    ElementData,
    PanelMessage,
    ProjectOrigin
  } from './utils/types'
  import { flattenElements, roomPath } from './utils/rooms'

  let graphData: any = $state(null)
  let currentProject: string | null = $state(null)
  // Set for a forked project: the project it was copied from
  let projectOrigin: ProjectOrigin | null = $state(null)
  // The room being viewed, or null for the main graph
  let activeRoom: string | null = $state(null)
  let roomCrumbs = $derived(roomPath(flattenElements(graphData), activeRoom))
  let audioSrc: string | null = $state(null)
  let audioTitle: string | null = $state(null)
  let showSpringEdges = $state(false)
  let showDetailedLabelsToggle = $state(false)
  let showEdgeLabels = $state(false)
  let chronologicalConstraint = $state(false)
  let hideEvolutionaryEdges = $state(false)
  let hideAllEdges = $state(false)
  let isPeekPressed = $state(false)
  let showDetailedLabels = $derived(showDetailedLabelsToggle || isPeekPressed)
  let selectedElementData: ElementData | null = $state(null)
  let actionPanelView: ActionPanelView = $state('none')
  let activeModelElement: any = $state(null)
  let activeAudioElement: ElementData | null = $state(null)
  let panelMessage: PanelMessage | null = $state(null)
  let allOperations: any[] = $state([])
  let engineState: EngineState | null = $state(null)
  let toolbarComponent: Toolbar

  let isWaitingForBackend = $state(true)
  // Backend restart detection
  let serverInstanceId: string | null = $state(null)
  let healthCheckInterval: number | null = null

  let hasDismissableJobs = $derived(
    $jobStore.some((job) => ['success', 'error', 'cancelled'].includes(job.status))
  )
  let hasCancellableJobs = $derived(
    $jobStore.some((job) => ['running', 'pending'].includes(job.status))
  )

  let knownJobIds = new SvelteSet<string>()
  let refreshGraphTimeout: number | null = null

  function debouncedRefreshGraphData(delayMs = 350): void {
    if (refreshGraphTimeout !== null) {
      window.clearTimeout(refreshGraphTimeout)
    }
    refreshGraphTimeout = window.setTimeout((): void => {
      refreshGraphTimeout = null
      void refreshGraphData()
    }, delayMs)
  }

  $effect((): (() => void) => {
    const unsub = jobStore.subscribe((jobs) => {
      let shouldRefresh = false

      for (const job of jobs) {
        if (!knownJobIds.has(job.id)) {
          knownJobIds.add(job.id)
        }
        if (job.status === 'success' && !job.result?.viewed) {
          shouldRefresh = true
          if (job.result) {
            job.result.viewed = true
          }
        }
      }

      if (shouldRefresh) {
        debouncedRefreshGraphData(350)
      }
    })
    return (): void => {
      unsub()
      if (refreshGraphTimeout !== null) {
        window.clearTimeout(refreshGraphTimeout)
      }
    }
  })

  onMount(async (): Promise<void> => {
    const ready = await initialHealthCheck()
    setupHealthCheckPolling()
    if (ready) {
      await restoreEngine()
      await loadOperations()
    }
  })

  // Reapplies the saved engine choice, which the backend forgets when it restarts
  async function restoreEngine(): Promise<void> {
    try {
      const restored = await window.api.restoreEngine()
      engineState =
        restored && !('conflict' in restored) ? restored : await window.api.getEngine()
    } catch (error: unknown) {
      console.error('Failed to restore engine choice:', error)
      const message = error instanceof Error ? error.message : String(error)
      panelMessage = { title: 'Engine Restore Failed', message }
    }
  }

  // Throws on failure so the engine menu can show the error in place
  async function handleSwitchEngine(settings: EngineSettings): Promise<void> {
    let result = await window.api.setEngine(settings)
    if ('conflict' in result) {
      const confirmed = window.confirm(
        `${result.unfinished_jobs} job(s) are still running. Their results will be lost if you switch engines now. Switch anyway?`
      )
      if (!confirmed) return
      result = await window.api.setEngine(settings, true)
      if ('conflict' in result) throw new Error(result.error)
    }
    engineState = result
    // Generative operations come from the active engine
    await loadOperations()
  }

  async function loadOperations(): Promise<void> {
    try {
      const res = await window.api.getOperations()
      if (res && res.success && Array.isArray(res.operations)) {
        allOperations = res.operations
      }
    } catch (error: unknown) {
      console.error('Failed to pre-load operations:', error)
    }
  }

  onDestroy((): void => {
    if (healthCheckInterval) {
      clearInterval(healthCheckInterval)
    }
  })

  async function initialHealthCheck(): Promise<boolean> {
    try {
      const status = await window.api.getHealth()
      if (status && status.status === 'healthy') {
        backendStatus.set(status)
        serverInstanceId = status.server_instance_id || null
        console.log('Backend status:', status)
        isWaitingForBackend = false
        return true
      }
    } catch {
      // Backend not yet ready; proceed to retry loop
    }

    const MAX_RETRIES = 20
    const RETRY_INTERVAL = 1500
    for (let i = 0; i < MAX_RETRIES; i++) {
      await new Promise((resolve): number => window.setTimeout(resolve, RETRY_INTERVAL))
      try {
        const status = await window.api.getHealth()
        if (status && status.status === 'healthy') {
          backendStatus.set(status)
          serverInstanceId = status.server_instance_id || null
          console.log(`Backend connected after ${i + 1} attempt(s):`, status)
          isWaitingForBackend = false
          return true
        }
      } catch {
        // Suppress noisy logs during normal startup retries
      }
    }
    isWaitingForBackend = false
    panelMessage = {
      title: 'Backend Connection Failed',
      message:
        'Could not connect to the backend server. Please ensure it is running and accessible.'
    }
    return false
  }

  function setupHealthCheckPolling(): void {
    if (healthCheckInterval) {
      clearInterval(healthCheckInterval)
    }
    healthCheckInterval = window.setInterval(async (): Promise<void> => {
      try {
        const status = await window.api.getHealth()
        if (status && status.status === 'healthy') {
          backendStatus.set(status)

          if (serverInstanceId && status.server_instance_id !== serverInstanceId) {
            console.warn('Backend has restarted. Resetting project state.')
            handleBackendRestart()
            void restoreEngine()
          }
          serverInstanceId = status.server_instance_id || null
        }
      } catch (error: unknown) {
        console.debug('Health check poll failed:', error)
      }
    }, 3000) // Poll every 3 seconds
  }

  function handleBackendRestart(): void {
    currentProject = null
    projectOrigin = null
    activeRoom = null
    graphData = null
    selectedElementData = null
    audioSrc = null
    audioTitle = null
    actionPanelView = 'none'
    $isCreatingNewProject = false

    // Close any selection process
    initiatorNodeStore.set(null)
    selectedForRemoval.set(null)
  }

  async function handleUpdateEmbeddings(): Promise<void> {
    try {
      await startEmbeddingUpdate()
    } catch (error: any) {
      console.error('Failed to update embeddings:', error)
      panelMessage = {
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
      panelMessage = {
        title: 'Update Labels Failed',
        message: error?.message || String(error)
      }
    }
  }

  async function handleRepairEdges(): Promise<void> {
    try {
      const response = await window.api.repairEdges()
      await refreshGraphData()

      panelMessage = {
        title: 'Repair Edges Successful',
        message: response?.message || 'Missing edges were successfully restored.',
        kind: 'success'
      }
    } catch (error: any) {
      console.error('Failed to repair edges:', error)
      panelMessage = {
        title: 'Repair Edges Failed',
        message: error?.message || String(error)
      }
    }
  }

  function closeActionPanel(): void {
    actionPanelView = 'none'
    initiatorNodeStore.set(null)
    contextStore.set(null)
    selectedForRemoval.set(null)
    selectedOperation.set(null)
    activeAudioElement = null
  }

  async function handleProjectLoad(data: {
    project_path?: string
    project_name?: string
  }): Promise<boolean> {
    try {
      const projectName = data.project_name || data.project_path?.split(/[/\\]/).pop()
      console.log(`Loading project: ${projectName}`)
      const loaded = await window.api.loadProject(data)
      await loadOperations()
      graphData = await window.api.getGraphData()
      currentProject = projectName || null
      projectOrigin = loaded.origin ?? null
      // Loading a project resets the backend to its main graph
      activeRoom = null
      if (data.project_path) {
        await window.api.addRecentProject(data.project_path)
      }
      console.log(`Successfully loaded project: ${projectName}`)
      return true
    } catch (error) {
      console.error('Failed to load project:', error)
      panelMessage = { title: 'Project Load Failed', message: error.message }
      return false
    }
  }

  async function handleProjectCreate(data: {
    project_path?: string
    project_name?: string
  }): Promise<void> {
    const projectName = data.project_name || data.project_path?.split(/[/\\]/).pop()
    console.log(`Creating project: ${projectName} at ${data.project_path}`)
    await window.api.createProject(data) // Pass the whole data object
    await loadOperations()
    graphData = await window.api.getGraphData()
    currentProject = projectName || null
    projectOrigin = null
    activeRoom = null
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
      panelMessage = { title: 'Add Source Failed', message: error.message || String(error) }
    }
  }

  async function handleExpandPath(pathNodeId: string): Promise<void> {
    try {
      const response = await window.api.expandPath(pathNodeId)
      if (response && response.job_id) {
        addJob('Expand Directory', { pathNodeId }, 'running', response.job_id)
        await pollJobStatus(response.job_id)
      }
      await refreshGraphData()
    } catch (error: any) {
      console.error('Error expanding path:', error)
      panelMessage = { title: 'Expand Path Failed', message: error.message || String(error) }
    }
  }

  async function handleRescanSource(sourceName: string): Promise<void> {
    try {
      const response = await window.api.rescanSource(sourceName)
      if (response && response.job_id) {
        addJob('Rescan Source', { sourceName }, 'running', response.job_id)
        await pollJobStatus(response.job_id)
      }
      await refreshGraphData()
    } catch (error: any) {
      console.error('Error rescanning source:', error)
      panelMessage = { title: 'Rescan Source Failed', message: error.message || String(error) }
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
        let dataArray: Uint8Array
        if (
          buffer &&
          typeof buffer === 'object' &&
          'data' in buffer &&
          Array.isArray((buffer as any).data)
        ) {
          dataArray = new Uint8Array((buffer as any).data)
        } else {
          dataArray = new Uint8Array(buffer)
        }
        const blob = new Blob([dataArray], { type: mimeType })
        audioSrc = URL.createObjectURL(blob)
        audioTitle = audioData.name
      } else {
        throw new Error('Failed to get audio file buffer.')
      }
    } catch (error) {
      console.error('Error getting audio file:', error)
      panelMessage = { title: 'Audio Load Failed', message: error.message }
      audioSrc = null
      audioTitle = null
    }
  }

  let imageSrc: string | null = $state(null)
  let imageTitle: string | null = $state(null)

  async function handleElementSelect(elementData: any): Promise<void> {
    selectedElementData = elementData
    panelMessage = null // Clear any existing errors when a new element is selected
    console.log(`Element selected: ${(selectedElementData?.name as string) || 'Unknown'}`)

    if (imageSrc && imageSrc.startsWith('blob:')) {
      URL.revokeObjectURL(imageSrc)
    }
    imageSrc = null
    imageTitle = null

    if (elementData && elementData.type === 'image') {
      try {
        const result = await window.api.getImageFile(elementData.id)
        if (result && result.buffer) {
          const { buffer, mimeType } = result
          let dataArray: Uint8Array
          if (
            buffer &&
            typeof buffer === 'object' &&
            'data' in buffer &&
            Array.isArray((buffer as any).data)
          ) {
            dataArray = new Uint8Array((buffer as any).data)
          } else {
            dataArray = new Uint8Array(buffer)
          }
          const blob = new Blob([dataArray], { type: mimeType })
          imageSrc = URL.createObjectURL(blob)
          imageTitle = elementData.name
        } else {
          throw new Error('Image data is empty or invalid.')
        }
      } catch (error: any) {
        console.error('Failed to get image file:', error)
        panelMessage = {
          title: 'Image Load Failed',
          message: error?.message || String(error)
        }
      }
    }
  }

  function handleImportGrating(modelData: any): void {
    initiatorNodeStore.set(modelData)
    actionPanelView = 'import-grating'
  }

  function handleBendModel(modelData: any): void {
    activeModelElement = modelData
    actionPanelView = 'bend'
  }

  function handleSelectOperation(op: any, initiatorNode: any, useContext = false): void {
    initiatorNodeStore.set(initiatorNode)
    selectedOperation.set(op)
    if (useContext) {
      contextStore.set(initiatorNode.context || null)
    } else {
      contextStore.set(null)
    }
    actionPanelView = 'operation'
  }

  function handleElementRemove(elementData: any): void {
    selectedForRemoval.set(elementData)
    actionPanelView = 'removal'
  }

  async function enterRoom(roomId: string | null): Promise<void> {
    try {
      // The backend places new elements in the viewed room
      await window.api.setViewRoom(roomId)
      activeRoom = roomId
    } catch (error: unknown) {
      console.error('Failed to change room:', error)
      const message = error instanceof Error ? error.message : String(error)
      panelMessage = { title: 'Changing Room Failed', message }
    }
  }

  async function handleBreakout(elementIds: string[]): Promise<void> {
    try {
      await window.api.createRoom(elementIds)
      await refreshGraphData()
    } catch (error: unknown) {
      console.error('Break out failed:', error)
      const message = error instanceof Error ? error.message : String(error)
      panelMessage = { title: 'Break Out Failed', message }
    }
  }

  async function handleDissolveRoom(roomId: string): Promise<void> {
    try {
      // Leaving a room that is about to disappear: go to the view it rejoins
      const crumbIndex = roomCrumbs.findIndex((crumb) => crumb.id === roomId)
      if (crumbIndex >= 0) {
        await enterRoom(crumbIndex > 0 ? roomCrumbs[crumbIndex - 1].id : null)
      }
      await window.api.dissolveRoom(roomId)
      await refreshGraphData()
    } catch (error: unknown) {
      console.error('Rejoin failed:', error)
      const message = error instanceof Error ? error.message : String(error)
      panelMessage = { title: 'Rejoin Failed', message }
    }
  }

  async function handleForkProject(elementIds: string[]): Promise<void> {
    let result: ForkResult
    try {
      result = await window.api.forkProject(elementIds)
    } catch (error: unknown) {
      console.error('Fork to project failed:', error)
      const message = error instanceof Error ? error.message : String(error)
      panelMessage = { title: 'Fork to Project Failed', message }
      return
    }

    // Switch to the fork right away; the original is one click back
    if (!(await handleProjectLoad({ project_path: result.project_path }))) return
    await toolbarComponent.refreshRecentProjects()
    // Spring edges are not carried over; rebuild them for the fork
    await handleUpdateEmbeddings()

    const converted = result.converted_exemplars.length
    if (converted) {
      panelMessage = {
        title: 'Forked to Project',
        message: `${converted} exemplar(s) were forked without their individual and are now plain audio. Their lineage is kept in their context.`,
        kind: 'info'
      }
    }
  }

  async function handleReturnToOrigin(): Promise<void> {
    if (!projectOrigin) return
    if (await handleProjectLoad({ project_path: projectOrigin.project_path })) {
      await toolbarComponent.refreshRecentProjects()
    }
  }

  function handleStartGrouping(nodeData: any): void {
    initiatorNodeStore.set(nodeData)
    actionPanelView = 'grouping'
  }

  async function handleExport(data: { names: string[]; isGroup?: boolean }): Promise<void> {
    try {
      let customPath: string | null = null
      if (!data.isGroup && data.names.length === 1) {
        const defaultName = data.names[0].endsWith('.wav') ? data.names[0] : `${data.names[0]}.wav`
        customPath = await window.api.selectSavePath(defaultName)
      } else {
        customPath = await window.api.selectDirectory()
      }
      if (!customPath) return // User cancelled

      const response = await window.api.exportAudio(data.names, customPath)
      let exportPath = response?.export_path || customPath
      if (response && response.job_id) {
        addJob(
          'Export Audio',
          { count: data.names.length, target: customPath },
          'running',
          response.job_id
        )
        const jobResult = (await pollJobStatus(response.job_id)) as {
          result?: { export_path?: string }
        }
        exportPath = jobResult?.result?.export_path || customPath
      }
      panelMessage = {
        title: 'Export Successful',
        message: `Successfully exported to: ${exportPath}`,
        kind: 'success'
      }
    } catch (error: unknown) {
      console.error('Export failed:', error)
      const message = error instanceof Error ? error.message : String(error)
      panelMessage = { title: 'Export Failed', message }
    }
  }

  async function handleChangeGroupMembership(
    _nodeId: string,
    oldGroupId: string | null,
    oldMembers: string[],
    newGroupId: string | null,
    newMembers: string[]
  ): Promise<void> {
    try {
      // Unlink from the old group if applicable
      if (oldGroupId) {
        await window.api.updateGroup(oldGroupId, oldMembers)
      }
      // Link to the new group if applicable
      if (newGroupId) {
        await window.api.updateGroup(newGroupId, newMembers)
      }
      await refreshGraphData()
    } catch (error: any) {
      console.error('Error changing group membership:', error)
      panelMessage = { title: 'Group Update Failed', message: error.message || String(error) }
    }
  }

  function handleGenerationError(error: PanelMessage): void {
    panelMessage = error
  }

  async function refreshGraphData(): Promise<void> {
    try {
      await loadOperations()
      graphData = await window.api.getGraphData()
      // The viewed room may have been removed
      if (activeRoom && roomCrumbs.length === 0) {
        await enterRoom(null)
      }
      console.log('Graph data refreshed')
    } catch (error) {
      console.error('Failed to get graph data:', error)
      panelMessage = { title: 'View Change Failed', message: error.message }
    }
  }

  function getInitiatorOutputType(): string | null {
    const node = $initiatorNodeStore
    if (!node) return null
    if (node.type === 'group') {
      const memberIds = node.member_ids || []
      if (memberIds.length > 0 && graphData && graphData.elements) {
        let elements: any[] = []
        if (Array.isArray(graphData.elements)) {
          elements = graphData.elements
        } else if (graphData.elements.nodes) {
          elements = graphData.elements.nodes
        }
        const firstMember = elements.find(
          (ele) => ele.data?.id === memberIds[0] || ele.id === memberIds[0]
        )
        if (firstMember) {
          const firstMemberData = firstMember.data || firstMember
          return firstMemberData.output_type || null
        }
      }
    }
    return node.output_type || null
  }

  function getActionPanelTitle(): string {
    if (actionPanelView === 'import-model') {
      return 'Import Model'
    }
    if (actionPanelView === 'import-grating') {
      return 'Import Grating'
    }
    if (actionPanelView === 'bend') {
      return 'Configure Model Bending'
    }
    if (actionPanelView === 'removal') {
      return 'Confirm Removal'
    }
    if (actionPanelView === 'grouping') {
      const initNode = $initiatorNodeStore
      if (initNode?.type === 'group') {
        const memberIds = (initNode as any).member_ids || []
        return memberIds.length === 0 ? 'Create Group' : 'Update Group'
      }
      return 'Create Group'
    }
    if (actionPanelView === 'operation') {
      if ($selectedOperation) {
        let initiatorType = $initiatorNodeStore?.type
        if ($contextStore) {
          if (
            $contextStore.init_audio ||
            $contextStore.source_audio ||
            $contextStore.source_audio_id
          ) {
            initiatorType = 'audio'
          } else if ($contextStore.init_latent) {
            initiatorType = 'latent'
          } else if ($contextStore.gratings && $contextStore.gratings.length > 0) {
            initiatorType = 'grating'
          } else {
            initiatorType = 'model'
          }
        }
        let displayName = $selectedOperation.name
        const outputType = getInitiatorOutputType()
        if ($selectedOperation.name === 'generate' && initiatorType === 'model') {
          if (outputType) {
            const capitalizedType = outputType.charAt(0).toUpperCase() + outputType.slice(1)
            displayName = `Generate ${capitalizedType}`
          } else {
            displayName = `Generate`
          }
        } else if (
          initiatorType &&
          $selectedOperation.context_overrides &&
          $selectedOperation.context_overrides[initiatorType]
        ) {
          const override = $selectedOperation.context_overrides[initiatorType]
          displayName = override.name || $selectedOperation.name
        }
        return `Operation: ${displayName.toUpperCase()}`
      }
      return 'Operation'
    }
    return 'Action'
  }

  async function handleToggleFavorite(data: any, isFavorite: boolean): Promise<void> {
    try {
      await window.api.updateElement(data.id, { favorite: isFavorite })
    } catch (error: any) {
      console.error('Error toggling favorite:', error)
      panelMessage = {
        title: 'Favorite Toggle Failed',
        message: error.message || String(error)
      }
    }
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

<svelte:window
  onkeydown={(e) => {
    if (e.key === 'Escape') selectionStore.cancelSelection()
    if (e.key === 'Alt') {
      e.preventDefault()
      isPeekPressed = true
    }
  }}
  onkeyup={(e) => {
    if (e.key === 'Alt') isPeekPressed = false
  }}
  onblur={() => (isPeekPressed = false)}
/>

<main class="container">
  {#if isWaitingForBackend}
    <div class="backend-wait-overlay">
      <div class="spinner" style="width: 32px; height: 32px; border-width: 4px;"></div>
      <p>Waiting for backend server...</p>
    </div>
  {/if}

  <Toolbar
    bind:this={toolbarComponent}
    engine={engineState}
    onswitchEngine={handleSwitchEngine}
    onprojectLoad={handleProjectLoad}
    onrefresh={refreshGraphData}
    onimportModel={() => (actionPanelView = 'import-model')}
    onaddExternalSource={handleAddExternalSource}
    onupdateEmbeddings={handleUpdateEmbeddings}
    onupdateLabels={handleUpdateLabels}
    onrepairEdges={handleRepairEdges}
    onreturnToOrigin={handleReturnToOrigin}
    onnavigateRoom={enterRoom}
    {currentProject}
    {projectOrigin}
    {roomCrumbs}
    bind:showSpringEdges
    bind:showDetailedLabels={showDetailedLabelsToggle}
    bind:showEdgeLabels
    bind:chronologicalConstraint
    bind:hideEvolutionaryEdges
    bind:hideAllEdges
  />

  {#if panelMessage}
    <ContentPanel title={panelMessage.title} onclose={() => (panelMessage = null)} position="left">
      <MessageView
        title={panelMessage.title}
        message={panelMessage.message}
        kind={panelMessage.kind}
      />
    </ContentPanel>
  {:else if actionPanelView !== 'none'}
    <ContentPanel title={getActionPanelTitle()} onclose={closeActionPanel} position="left">
      {#if actionPanelView === 'import-model'}
        <ImportModelView
          onclose={closeActionPanel}
          onrefresh={() => {
            closeActionPanel()
            refreshGraphData()
          }}
          onError={(error) => {
            console.error('Import error:', error)
            closeActionPanel()
            panelMessage = error
          }}
        />
      {:else if actionPanelView === 'import-grating'}
        <ImportGratingView
          onclose={closeActionPanel}
          onrefresh={() => {
            closeActionPanel()
            refreshGraphData()
          }}
          onError={(error) => {
            console.error('Grating import error:', error)
            closeActionPanel()
            panelMessage = error
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
            panelMessage = error
          }}
        />
      {:else if actionPanelView === 'grouping'}
        <GroupingView
          onclose={closeActionPanel}
          onrefresh={() => {
            closeActionPanel()
            refreshGraphData()
          }}
          onerror={(error) => {
            console.error('Grouping error:', error)
            closeActionPanel()
            panelMessage = error
          }}
        />
      {:else if actionPanelView === 'operation'}
        <OperationView onClose={closeActionPanel} onError={handleGenerationError} />
      {:else if actionPanelView === 'bend'}
        <BendingView
          modelElement={activeModelElement}
          onclose={closeActionPanel}
          onrefresh={() => {
            closeActionPanel()
            refreshGraphData()
          }}
          onError={(error) => {
            console.error('Bending setup error:', error)
            closeActionPanel()
            panelMessage = error
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
        {#snippet header_actions()}
          {#if hasCancellableJobs}
            <button class="clear-btn" onclick={cancelAllJobs}>Cancel All</button>
          {/if}
          {#if hasDismissableJobs}
            <button class="clear-btn" onclick={clearDismissableJobs}>Clear Completed</button>
          {/if}
        {/snippet}
      </ContentPanel>
    {/if}

    {#if selectedElementData}
      <ContentPanel
        title={(selectedElementData.name as string) ||
          (selectedElementData.alias as string) ||
          'Element Details'}
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
    operations={allOperations}
    onaudioSelect={handleAudioSelect}
    onimportGrating={handleImportGrating}
    onbendModel={handleBendModel}
    onnodeSelect={handleElementSelect}
    onexport={handleExport}
    onedgeSelect={handleElementSelect}
    onElementRemove={handleElementRemove}
    onbreakout={handleBreakout}
    onforkProject={handleForkProject}
    onenterRoom={enterRoom}
    ondissolveRoom={handleDissolveRoom}
    {activeRoom}
    onstartGrouping={handleStartGrouping}
    onsavePositions={handleSavePositions}
    onexpandPath={handleExpandPath}
    onrescanSource={handleRescanSource}
    ontoggleFavorite={handleToggleFavorite}
    {showDetailedLabels}
    {showEdgeLabels}
    {chronologicalConstraint}
    {hideEvolutionaryEdges}
    {hideAllEdges}
    onchangeGroupMembership={handleChangeGroupMembership}
    onselectOperation={handleSelectOperation}
    onrefresh={refreshGraphData}
  />
  {#if audioSrc}
    <AudioPlayer src={audioSrc} title={audioTitle} onclose={() => (audioSrc = null)} />
  {/if}
  {#if imageSrc}
    <div class="image-preview-card">
      <div class="preview-header">
        <span>{imageTitle}</span>
        <button
          class="close-btn"
          onclick={() => {
            if (imageSrc && imageSrc.startsWith('blob:')) {
              URL.revokeObjectURL(imageSrc)
            }
            imageSrc = null
          }}>×</button
        >
      </div>
      <div class="preview-body">
        <img src={imageSrc} alt={imageTitle} />
      </div>
    </div>
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

  .image-preview-card {
    position: absolute;
    bottom: 1rem;
    left: 50%;
    transform: translateX(-50%);
    width: 256px;
    background: var(--color-background-glass-2, rgba(20, 20, 25, 0.7));
    backdrop-filter: blur(12px);
    border: 1px solid var(--color-overlay-border-primary, rgba(255, 255, 255, 0.1));
    border-radius: 0.75rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    z-index: 1000;
  }

  .preview-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.75rem;
    background: rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 0.8rem;
    color: var(--color-overlay-text, #fff);
  }

  .preview-header span {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-right: 0.5rem;
  }

  .close-btn {
    background: transparent !important;
    border: none !important;
    color: var(--color-text-overlay-secondary, #ccc) !important;
    font-size: 1.2rem !important;
    padding: 0 !important;
    min-width: 20px !important;
    min-height: 20px !important;
    cursor: pointer;
    line-height: 1;
  }

  .close-btn:hover {
    color: #fff !important;
  }

  .preview-body {
    padding: 0.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .preview-body img {
    max-width: 100%;
    max-height: 240px;
    border-radius: 0.375rem;
    object-fit: contain;
  }
</style>
