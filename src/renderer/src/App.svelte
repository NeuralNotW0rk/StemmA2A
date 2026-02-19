<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  import ParameterGraph from './components/graph/ParameterGraph.svelte'
  import Toolbar from './components/Toolbar.svelte'
  import AudioPlayer from './components/AudioPlayer.svelte'
  import Sidebar from './components/Sidebar.svelte'

  let graphData: any = null
  let currentProject: string | null = null
  let viewMode: 'batch' | 'cluster' = 'batch'
  let parameterGraph: ParameterGraph
  let audioSrc: string | null = null
  let audioTitle: string | null = null
  let selectedNodeData: Record<string, any> | null = null

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
      console.log(`Failed to load project: ${projectPath}. Error: ${error}`)
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
      console.log(`Failed to create project: ${projectPath}. Error: ${error}`)
    }
  }

  async function handleViewModeChange(mode: 'batch' | 'cluster'): Promise<void> {
    viewMode = mode
    try {
      graphData = await window.api.getGraphData(viewMode)
    } catch (error) {
      console.error('Failed to get graph data:', error)
      console.log(`Failed to get graph data for view mode ${viewMode}. Error: ${error}`)
    }
  }

  async function handleAudioSelect(audioData: any): Promise<void> {
    console.log(`Audio selected: ${audioData.name}`)

    // Revoke the old blob URL if it exists to prevent memory leaks
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
        console.error('Failed to get audio file buffer.')
        audioSrc = null
        audioTitle = null
      }
    } catch (error) {
      console.error('Error getting audio file:', error)
      audioSrc = null
      audioTitle = null
    }
  }

  function handleNodeSelect(nodeData: any): void {
    selectedNodeData = nodeData
    console.log(`Node selected: ${selectedNodeData?.name || 'Unknown'}`)
  }

  async function refreshGraphData(): Promise<void> {
    try {
      graphData = await window.api.getGraphData(viewMode)
      console.log(`Graph data refreshed for view mode ${viewMode}`)
    } catch (error) {
      console.error('Failed to get graph data:', error)
      console.log(`Failed to get graph data for view mode ${viewMode}. Error: ${error}`)
    }
  }
</script>

<main class="container">
  <Toolbar
    onprojectLoad={handleProjectLoad}
    onprojectCreate={handleProjectCreate}
    onviewModeChange={handleViewModeChange}
    onrefresh={refreshGraphData}
    {currentProject}
    {viewMode}
  />
  {#if selectedNodeData}
    <Sidebar {selectedNodeData} onclose={() => (selectedNodeData = null)} />
  {/if}
  <ParameterGraph
    bind:this={parameterGraph}
    {graphData}
    {viewMode}
    onaudioSelect={handleAudioSelect}
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
</style>
