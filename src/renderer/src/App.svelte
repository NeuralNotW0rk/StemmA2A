<script lang="ts">
  import AudioGraph from './components/AudioGraph.svelte'
  import Toolbar from './components/Toolbar.svelte'

  let graphData: any = null
  let currentProject: string | null = null

  async function handleProjectLoad(event: CustomEvent<{ projectPath: string }>): Promise<void> {
    const projectPath = event.detail.projectPath
    try {
      graphData = await window.api.loadProjectAndGetData(projectPath)
      currentProject = projectPath
      await window.api.addRecentProject(projectPath)
    } catch (error) {
      console.error('Failed to load project:', error)
      // Optionally, show an error message to the user
    }
  }
</script>

<main class="container">
  <Toolbar on:projectLoad={handleProjectLoad} {currentProject} />
  <AudioGraph {graphData} />
</main>

<style>
  .container {
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
