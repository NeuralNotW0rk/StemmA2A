<!-- src/renderer/src/components/AudioGraph.svelte -->
<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte'
  import cytoscape from 'cytoscape'
  import fcose from 'cytoscape-fcose'
  import cxtmenu from 'cytoscape-cxtmenu'
  import expandCollapse from 'cytoscape-expand-collapse'

  export let graphData: any = null

  const dispatch = createEventDispatcher()

  let graphContainer: HTMLElement
  let cy: cytoscape.Core | null = null
  let isInitialized = false

  // Graph styling
  const graphStyle = [
    {
      selector: 'node',
      style: {
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': '12px',
        'font-weight': 'bold',
        'color': '#ffffff',
        'text-outline-width': 2,
        'text-outline-color': '#000000',
        'text-wrap': 'wrap',
        'text-max-width': '80px',
        'width': 60,
        'height': 60,
        'overlay-padding': '4px'
      }
    },
    {
      selector: 'node[type="model"]',
      style: {
        'background-color': '#9333ea',
        'shape': 'round-rectangle',
        'width': 80,
        'height': 50
      }
    },
    {
      selector: 'node[type="audio"]',
      style: {
        'background-color': '#06b6d4',
        'shape': 'ellipse'
      }
    },
    {
      selector: 'node[type="batch"]',
      style: {
        'background-color': '#374151',
        'shape': 'round-rectangle',
        'width': 100,
        'height': 60,
        'border-width': 2,
        'border-color': '#6b7280'
      }
    },
    {
      selector: 'node[type="external"]',
      style: {
        'background-color': '#059669',
        'shape': 'round-diamond'
      }
    },
    {
      selector: 'node:selected',
      style: {
        'border-width': 3,
        'border-color': '#f59e0b'
      }
    },
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#6b7280',
        'target-arrow-color': '#6b7280',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier'
      }
    },
    {
      selector: 'edge[type="variation"]',
      style: {
        'line-style': 'dashed',
        'line-color': '#06b6d4'
      }
    },
    {
      selector: 'edge[type="generation"]',
      style: {
        'line-color': '#9333ea'
      }
    }
  ]

  // Layout configuration
  const layoutConfig = {
    name: 'fcose',
    quality: 'default',
    randomize: false,
    animate: true,
    animationDuration: 1000,
    fit: true,
    padding: 30,
    nodeDimensionsIncludeLabels: true,
    uniformNodeDimensions: false,
    packComponents: true,
    nodeRepulsion: () => 8000,
    idealEdgeLength: () => 200,
    edgeElasticity: () => 0.1,
    nestingFactor: 0.1,
    numIter: 2500,
    tile: true
  }

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
      layoutBy: layoutConfig,
      fisheye: false,
      animate: true,
      undoable: false
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
          select: (ele: any) => console.log('Model info:', ele.data())
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

  function applyLayout() {
    if (!cy) return
    cy.layout(layoutConfig).run()
  }

  function updateGraph() {
    if (!cy || !graphData) return

    try {
      // Store current positions
      const positions: { [key: string]: cytoscape.Position } = {}
      cy.nodes().forEach(node => {
        positions[node.id()] = node.position()
      })

      // Clear existing elements
      cy.elements().remove()

      // Add new elements
      if (graphData.elements) {
        cy.add(graphData.elements)

        // Restore positions for existing nodes
        cy.nodes().forEach(node => {
          if (positions[node.id()]) {
            node.position(positions[node.id()])
          }
        })

        // Apply layout if significant changes
        const needsLayout = Object.keys(positions).length === 0 || 
                           cy.nodes().length !== Object.keys(positions).length

        if (needsLayout) {
          applyLayout()
        }

        // Setup batch expansion state
        cy.nodes('[type="batch"]').forEach(node => {
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
</script>

<div class="graph-wrapper">
  <div bind:this={graphContainer} class="graph-container"></div>
  
  <div class="graph-controls">
    <button on:click={applyLayout} title="Reorganize layout">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
      </svg>
    </button>
    
    <button on:click={() => cy?.fit()} title="Fit to screen">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
      </svg>
    </button>
    
    <button on:click={() => cy?.center()} title="Center view">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
      </svg>
    </button>
  </div>

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
    height: 100%;
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

  .graph-controls {
    position: absolute;
    top: 1rem;
    right: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    z-index: 100;
  }

  .graph-controls button {
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 0.5rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
    backdrop-filter: blur(10px);
  }

  .graph-controls button:hover {
    background: rgba(0, 0, 0, 0.8);
    border-color: rgba(255, 255, 255, 0.3);
    transform: translateY(-1px);
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