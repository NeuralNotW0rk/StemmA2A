<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  import AudioGraph from './components/graph/ParameterGraph.svelte'
  import Toolbar from './components/Toolbar.svelte'
  import AudioPlayer from './components/AudioPlayer.svelte'
  import Sidebar from './components/Sidebar.svelte'

  let graphData: any = null
  let currentProject: string | null = null
  let viewMode: 'batch' | 'cluster' = 'batch'
  let audioGraph: AudioGraph
  let audioSrc: string | null = null
  let audioTitle: string | null = null
  let selectedNodeData: Record<string, any> | null = null

  async function handleProjectLoad(event: CustomEvent<{ projectPath: string }>): Promise<void> {
    const projectPath = event.detail.projectPath
    try {
      await window.api.logMessage(`Loading project: ${projectPath}`)
      await window.api.loadProject(projectPath)
      graphData = await window.api.getGraphData(viewMode)
      currentProject = projectPath
      await window.api.addRecentProject(projectPath)
      await window.api.logMessage(`Successfully loaded project: ${projectPath}`)
    } catch (error) {
      console.error('Failed to load project:', error)
      await window.api.logMessage(`Failed to load project: ${projectPath}. Error: ${error}`)
    }
  }

  async function handleViewModeChange(event: CustomEvent<'batch' | 'cluster'>): Promise<void> {
    viewMode = event.detail
    try {
      graphData = await window.api.getGraphData(viewMode)
    } catch (error) {
      console.error('Failed to get graph data:', error)
      await window.api.logMessage(
        `Failed to get graph data for view mode ${viewMode}. Error: ${error}`
      )
    }
  }

  async function handleAudioSelect(event: CustomEvent<any>): Promise<void> {
    const audioData = event.detail
    await window.api.logMessage(`Audio selected: ${audioData.name}`)
    try {
      const dataUrl = await window.api.getAudioFile(audioData.name)
      if (dataUrl) {
        audioSrc = dataUrl
        audioTitle = audioData.alias
      } else {
        console.error('Failed to get audio data URL.')
      }
    } catch (error) {
      console.error('Error getting audio file:', error)
    }
  }

  function handleNodeSelect(event: CustomEvent<any>): void {
    selectedNodeData = event.detail
    window.api.logMessage(`Node selected: ${selectedNodeData?.name || 'Unknown'}`)
  }

  async function refreshGraphData(): Promise<void> {
    try {
      graphData = await window.api.getGraphData(viewMode)
      await window.api.logMessage(`Graph data refreshed for view mode ${viewMode}`)
    } catch (error) {
      console.error('Failed to get graph data:', error)
      await window.api.logMessage(
        `Failed to get graph data for view mode ${viewMode}. Error: ${error}`
      )
    }
  }
</script>

<main class="container">
  <Toolbar
    on:projectLoad={handleProjectLoad}
    on:viewModeChange={handleViewModeChange}
    on:refresh={refreshGraphData}
    {currentProject}
    {viewMode}
  />
  {#if selectedNodeData}
    <Sidebar {selectedNodeData} on:close={() => (selectedNodeData = null)} />
  {/if}
  <AudioGraph
    bind:this={audioGraph}
    {graphData}
    {viewMode}
    on:audioSelect={handleAudioSelect}
    on:nodeSelect={handleNodeSelect}
  />
  {#if audioSrc}
    <AudioPlayer src={audioSrc} title={audioTitle} on:close={() => (audioSrc = null)} />
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
