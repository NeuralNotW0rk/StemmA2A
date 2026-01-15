<!-- src/renderer/src/components/AudioGraph.svelte -->
<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte'
  import cytoscape from 'cytoscape'
  import fcose from 'cytoscape-fcose'
  import cxtmenu from 'cytoscape-cxtmenu'
  import expandCollapse from 'cytoscape-expand-collapse'
  import graphStyle from './Style.js'
  import layoutConfig from './Layout.js'
  import expandCollapseOptions from './Options.js'

  export let graphData: any = null
  export let viewMode: 'batch' | 'cluster' = 'batch'

  const dispatch = createEventDispatcher()
  let graphContainer: HTMLElement
  let cy: cytoscape.Core | null = null
  let isInitialized = false

  onMount(() => {
    initializeGraph()
  })

  onDestroy(() => {
    if (cy) {
      cy.destroy()
    }
  })

  function initializeGraph() {
    // Register extensions
    cytoscape.use(fcose)
    cytoscape.use(cxtmenu)
    cytoscape.use(expandCollapse)

    // Initialize cytoscape
    cy = cytoscape({
      container: graphContainer,
      style: graphStyle,
      layout: { name: 'grid' }, // Temporary layout
      wheelSensitivity: 0.2,
      maxZoom: 3,
      minZoom: 0.1
    })

    // Initialize expand/collapse
    cy.expandCollapse({
      ...expandCollapseOptions,
      layoutBy: layoutConfig
    })

    setupContextMenus()
    setupEventListeners()
    isInitialized = true

    // Load initial data if available
    if (graphData) {
      updateGraph()
    }
  }

  function setupContextMenus() {
    if (!cy) return

    // Core context menu (right-click on empty space)
    cy.cxtmenu({
      selector: 'core',
      commands: [
        {
          content: 'Tidy Layout',
          select: () => applyLayout()
        },
        {
          content: 'Import Model',
          select: () => dispatch('importModel')
        },
        {
          content: 'Add External Source',
          select: () => dispatch('addExternalSource')
        },
        {
          content: 'Refresh',
          select: () => dispatch('refresh')
        }
      ]
    })

    // Model node context menu
    cy.cxtmenu({
      selector: 'node[type="model"]',
      commands: [
        {
          content: 'Generate Audio',
          select: (ele: any) => {
            const modelData = ele.data()
            dispatch('modelSelect', modelData)
          }
        },
        {
          content: 'Model Info',
          select: (ele: any) => dispatch('modelSelect', ele.data())
        }
      ]
    })

    // Audio node context menu
    cy.cxtmenu({
      selector: 'node[type="audio"]',
      commands: [
        {
          content: 'Play/Select',
          select: (ele: any) => {
            const audioData = ele.data()
            dispatch('audioSelect', audioData)
          }
        },
        {
          content: 'Create Variation',
          select: (ele: any) => {
            const audioData = ele.data()
            dispatch('variate', { source_name: audioData.name })
          }
        },
        {
          content: 'Export',
          select: (ele: any) => {
            const audioData = ele.data()
            dispatch('export', { names: [audioData.name] })
          }
        }
      ]
    })

    // Batch node context menu
    cy.cxtmenu({
      selector: 'node[type="batch"]',
      commands: [
        {
          content: 'Expand/Collapse',
          select: (ele: any) => {
            const api = cy?.expandCollapse('get')
            if (api) {
              if (ele.data('isExpanded')) {
                api.collapse(ele)
              } else {
                api.expand(ele)
              }
            }
          }
        },
        {
          content: 'Export Batch',
          select: (ele: any) => {
            const batchData = ele.data()
            dispatch('export', { names: batchData.children || [] })
          }
        }
      ]
    })

    // External source context menu
    cy.cxtmenu({
      selector: 'node[type="external"]',
      commands: [
        {
          content: 'Rescan',
          select: (ele: any) => {
            const sourceData = ele.data()
            dispatch('rescanSource', sourceData.name)
          }
        }
      ]
    })
  }

  function setupEventListeners() {
    if (!cy) return

    // Audio node selection
    cy.on('tap', 'node[type="audio"]', (evt) => {
      console.log('Audio node tapped:', evt.target.data());
      const audioData = evt.target.data()
      dispatch('audioSelect', audioData)
    })

    // Model node selection
    cy.on('tap', 'node[type="model"]', (evt) => {
      const modelData = evt.target.data()
      dispatch('modelSelect', modelData)
    })

    // Double-click to fit
    cy.on('dblclick', 'core', () => {
      cy?.fit()
    })

    // Keyboard shortcuts
    cy.on('keydown', (evt) => {
      if (evt.originalEvent.key === 'r') {
        applyLayout()
      }
    })
  }

    function applyLayout(randomize = false) {

      if (!cy) return

      const scale = 500

  

      var fixedNodeConstraint = []

      if (viewMode === 'cluster') {

        fixedNodeConstraint = cy.$('node[type="audio"]').map((ele) => {

          return {

            nodeId: ele.data('id'),

            position: { x: ele.data('tsne_1') * scale, y: ele.data('tsne_2') * scale }

          }

        })

      }

  

      // Workaround tiling issues by temporarily removing audio source edges

      var audioSourceEdges = cy.edges('[type="audio_source"]').remove()

  

      // Create and run layout

      var layout = cy.layout({

        ...layoutConfig,

        randomize,

        fixedNodeConstraint,

        tilingCompareBy: (nodeId1, nodeId2) => {

          if (cy.$id(nodeId1).data('type') === 'audio' && cy.$id(nodeId2).data('type') === 'audio') {

            return cy.$id(nodeId1).data('batch_index') - cy.$id(nodeId2).data('batch_index')

          }

          return 0

        }

      })

      layout.run()

  

      // Restore removed elements

      audioSourceEdges.restore()

    }

  function updateGraph() {
    if (!cy || !graphData) return

    try {
      // Store current positions
      const positions: { [key: string]: cytoscape.Position } = {}
      cy.nodes().forEach((node) => {
        positions[node.id()] = node.position()
      })

      // Clear existing elements
      cy.elements().remove()

      // Add new elements
      if (graphData.elements) {
        cy.add(graphData.elements)

        // Restore positions for existing nodes, but only in 'batch' mode
        if (viewMode === 'batch') {
          cy.nodes().forEach((node) => {
            if (positions[node.id()]) {
              node.position(positions[node.id()])
            }
          })
        }

        // Apply layout if significant changes
        const needsLayout =
          Object.keys(positions).length === 0 || cy.nodes().length !== Object.keys(positions).length

        if (needsLayout) {
          applyLayout()
        }

        // Setup batch expansion state
        cy.nodes('[type="batch"]').forEach((node) => {
          node.data('isExpanded', true)
        })
      }
    } catch (error) {
      console.error('Error updating graph:', error)
    }
  }

  // React to data changes
  $: if (isInitialized && graphData) {
    updateGraph()
  }

  export function reorganizeLayout() {
    applyLayout()
  }

  export function fitView() {
    cy?.fit()
  }

  export function centerView() {
    cy?.center()
  }
</script>

<div class="graph-wrapper">
  <div bind:this={graphContainer} class="graph-container"></div>

  <div class="graph-legend">
    <div class="legend-item">
      <div class="legend-color model"></div>
      <span>Models</span>
    </div>
    <div class="legend-item">
      <div class="legend-color audio"></div>
      <span>Audio</span>
    </div>
    <div class="legend-item">
      <div class="legend-color batch"></div>
      <span>Batches</span>
    </div>
    <div class="legend-item">
      <div class="legend-color external"></div>
      <span>External</span>
    </div>
  </div>
</div>

<style>
  .graph-wrapper {
    position: relative;
    width: 100%;
    flex-grow: 1;
    min-height: 0;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  }

  .graph-container {
    width: 100%;
    height: 100%;
    cursor: grab;
  }

  .graph-container:active {
    cursor: grabbing;
  }

  .graph-legend {
    position: absolute;
    bottom: 1rem;
    left: 1rem;
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 0.5rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    backdrop-filter: blur(10px);
    z-index: 100;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: white;
  }

  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }

  .legend-color.model {
    background-color: #9333ea;
  }

  .legend-color.audio {
    background-color: #06b6d4;
  }

  .legend-color.batch {
    background-color: #374151;
    border: 1px solid #6b7280;
  }

  .legend-color.external {
    background-color: #059669;
  }
</style>
