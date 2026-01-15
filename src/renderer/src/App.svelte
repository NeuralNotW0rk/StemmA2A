<script lang="ts">
  import AudioGraph from './components/graph/ParameterGraph.svelte'
  import Toolbar from './components/Toolbar.svelte'
  import GraphControls from './components/graph/GraphControls.svelte'

  let graphData: any = null
  let currentProject: string | null = null
  let viewMode: 'batch' | 'cluster' = 'batch'
  let audioGraph: AudioGraph

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
      // Optionally, show an error message to the user
    }
  }

  async function handleViewModeChange(event: CustomEvent<'batch' | 'cluster'>) {
    viewMode = event.detail
    try {
      graphData = await window.api.getGraphData(viewMode)
    } catch (error) {
      console.error('Failed to get graph data:', error)
      await window.api.logMessage(`Failed to get graph data for view mode ${viewMode}. Error: ${error}`)
    }
  }
</script>

<main class="container">
  <Toolbar
    on:projectLoad={handleProjectLoad}
    on:viewModeChange={handleViewModeChange}
    {currentProject}
    {viewMode}
  />
  <AudioGraph bind:this={audioGraph} {graphData} {viewMode} />
  <GraphControls
    on:reorganize={() => audioGraph.reorganizeLayout()}
    on:fit={() => audioGraph.fitView()}
    on:center={() => audioGraph.centerView()}
  />
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
