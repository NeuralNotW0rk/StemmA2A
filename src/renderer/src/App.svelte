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

  type ActionPanelView = 'generation' | 'import-model' | 'none'

  let graphData: any = $state(null)
  let currentProject: string | null = $state(null)
  let viewMode: 'batch' | 'cluster' = $state('batch')
  let audioSrc: string | null = $state(null)
  let audioTitle: string | null = $state(null)
  let selectedNodeData: Record<string, any> | null = $state(null)
  let generationNode: any = $state(null)
  let actionPanelView: ActionPanelView = $state('none')
  let errorInInfoPanel: { title: string; message: string } | null = $state(null)

  function closeActionPanel() {
    actionPanelView = 'none'
    generationNode = null
  }

  async function handleProjectLoad(data: { projectPath: string }): Promise<void> {
    const projectPath = data.projectPath
    try {
      console.log(`Loading project: ${projectPath}`)
      await window.api.loadProject(projectPath)
      graphData = await window.api.getGraphData(viewMode)
      currentProject = projectPath
      await window.api.addRecentProject(projectPath)
      console.log(`Successfully loaded project: ${projectPath}`)
    } catch (error) {
      console.error('Failed to load project:', error)
      errorInInfoPanel = { title: 'Project Load Failed', message: error.message }
    }
  }

  async function handleProjectCreate(data: { projectPath: string }): Promise<void> {
    const projectPath = data.projectPath
    try {
      console.log(`Creating project: ${projectPath}`)
      await window.api.createProject(projectPath)
      graphData = await window.api.getGraphData(viewMode)
      currentProject = projectPath
      await window.api.addRecentProject(projectPath)
      console.log(`Successfully created project: ${projectPath}`)
    } catch (error) {
      console.error('Failed to create project:', error)
      errorInInfoPanel = { title: 'Project Create Failed', message: error.message }
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
      const result = await window.api.getAudioFile(audioData.name)
      if (result && result.buffer) {
        const { buffer, mimeType } = result
        const blob = new Blob([new Uint8Array(buffer)], { type: mimeType })
        audioSrc = URL.createObjectURL(blob)
        audioTitle = audioData.alias
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

  function handleNodeSelect(nodeData: any): void {
    selectedNodeData = nodeData
    errorInInfoPanel = null // Clear any existing errors when a new node is selected
    console.log(`Node selected: ${selectedNodeData?.name || 'Unknown'}`)
  }

  function handleModelSelect(modelData: any): void {
    generationNode = modelData
    actionPanelView = 'generation'
  }

  function handleAudioNodeSelectForGeneration(audioData: any): void {
    generationNode = audioData
    actionPanelView = 'generation'
  }

  async function refreshGraphData(): Promise<void> {
    try {
      graphData = await window.api.getGraphData(viewMode)
      console.log(`Graph data refreshed for view mode ${viewMode}`)
    } catch (error) {
      console.error('Failed to get graph data:', error)
      errorInInfoPanel = { title: 'Graph Refresh Failed', message: error.message }
    }
  }

  function getActionPanelTitle(): string {
    if (actionPanelView === 'import-model') {
      return 'Import Model'
    }
    if (actionPanelView === 'generation' && generationNode) {
      return generationNode.type === 'model' ? 'Generate' : 'Variation'
    }
    return 'Action'
  }
</script>

<main class="container">
  <Toolbar
    onprojectLoad={handleProjectLoad}
    onprojectCreate={handleProjectCreate}
    onviewModeChange={handleViewModeChange}
    onrefresh={refreshGraphData}
    onimportModel={() => (actionPanelView = 'import-model')}
    {currentProject}
    {viewMode}
  />

  {#if errorInInfoPanel}
    <ContentPanel
      title={errorInInfoPanel.title}
      onclose={() => (errorInInfoPanel = null)}
      position="left"
    >
      <ErrorView title={errorInInfoPanel.title} message={errorInInfoPanel.message} />
    </ContentPanel>
  {:else if selectedNodeData}
    <ContentPanel
      title={selectedNodeData.alias || selectedNodeData.name || 'Element Details'}
      onclose={() => (selectedNodeData = null)}
      position="right"
    >
      <ElementInfoView {selectedNodeData} />
    </ContentPanel>
  {/if}

  {#if actionPanelView !== 'none'}
    <ContentPanel title={getActionPanelTitle()} onclose={closeActionPanel} position="left">
      {#if actionPanelView === 'generation'}
        <GenerationView
          node={generationNode}
          onClose={closeActionPanel}
          onGenerate={(result) => {
            console.log('Generation result:', result)
            closeActionPanel()
            refreshGraphData()
          }}
          onError={(error) => {
            console.error('Generation error:', error)
            closeActionPanel()
            errorInInfoPanel = error
          }}
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
      {/if}
    </ContentPanel>
  {/if}

  <ParameterGraph
    {graphData}
    {viewMode}
    onaudioSelect={handleAudioSelect}
    onmodelSelect={handleModelSelect}
    onaudioNodeSelectForGeneration={handleAudioNodeSelectForGeneration}
    onnodeSelect={handleNodeSelect}
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
