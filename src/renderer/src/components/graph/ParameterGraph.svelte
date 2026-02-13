<!-- src/renderer/src/components/ParameterGraph.svelte -->
<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  import { onMount, onDestroy } from 'svelte'
  import cytoscape, { type EventObject, type NodeSingular } from 'cytoscape'
  import fcose from 'cytoscape-fcose'
  import cxtmenu from 'cytoscape-cxtmenu'
  import expandCollapse from 'cytoscape-expand-collapse'
  import graphStyle from './Style'
  import layoutConfig from './Layout'
  import expandCollapseOptions from './Options'
  import GraphControls from './GraphControls.svelte'
  import GraphLegend from './GraphLegend.svelte'

  interface Props {
    graphData?: { elements: cytoscape.ElementDefinition[] } | null
    viewMode?: 'batch' | 'cluster'
    onmodelSelect?: (data: any) => void
    onaudioSelect?: (data: any) => void
    onvariate?: (data: { source_name: string }) => void
    onexport?: (data: { names: string[] }) => void
    onrescanSource?: (name: string) => void
    onnodeSelect?: (data: any) => void
  }

  let {
    graphData = null,
    viewMode = 'batch',
    onmodelSelect,
    onaudioSelect,
    onvariate,
    onexport,
    onrescanSource,
    onnodeSelect
  }: Props = $props()

  let graphContainer: HTMLElement | undefined = $state()
  let cy: cytoscape.Core | null = $state(null)
  let isInitialized = $state(false)

  onMount(() => {
    initializeGraph()
  })

  onDestroy(() => {
    if (cy) {
      cy.destroy()
    }
  })

  function initializeGraph(): void {
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

  function setupContextMenus(): void {
    if (!cy) return

    // Model node context menu
    cy.cxtmenu({
      selector: 'node[type="model"]',
      commands: [
        {
          content: 'Generate Audio',
          select: (ele: any) => {
            const modelData = ele.data()
            onmodelSelect?.(modelData)
          }
        },
        {
          content: 'Model Info',
          select: (ele: any) => onmodelSelect?.(ele.data())
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
            onaudioSelect?.(audioData)
          }
        },
        {
          content: 'Create Variation',
          select: (ele: any) => {
            const audioData = ele.data()
            onvariate?.({ source_name: audioData.name })
          }
        },
        {
          content: 'Export',
          select: (ele: any) => {
            const audioData = ele.data()
            onexport?.({ names: [audioData.name] })
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
            onexport?.({ names: batchData.children || [] })
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
            onrescanSource?.(sourceData.name)
          }
        }
      ]
    })
  }

  function setupEventListeners(): void {
    if (!cy) return

    cy.on('tap', 'node, edge', (evt: EventObject) => {
      const elementData = evt.target.data()
      onnodeSelect?.(elementData) // Dispatch for sidebar

      // If the tapped element is a node and its type is 'audio'
      if (evt.target.isNode() && elementData.type === 'audio') {
        onaudioSelect?.(elementData) // Dispatch for audio player
      }
    })

    // Double-click to fit
    cy.on('dblclick', 'core', () => {
      cy?.fit()
    })

    // Keyboard shortcuts
    cy.on('keydown', (evt: any) => {
      if (evt.originalEvent.key === 'r') {
        applyLayout()
      }
    })
  }

  function applyLayout(randomize = false): void {
    if (!cy) return

    const scale = 500

    let fixedNodeConstraint: { nodeId: string; position: { x: number; y: number } }[] = []

    if (viewMode === 'cluster') {
      fixedNodeConstraint = cy.$('node[type="audio"]').map((ele: NodeSingular) => {
        return {
          nodeId: ele.data('id'),
          position: { x: ele.data('tsne_1') * scale, y: ele.data('tsne_2') * scale }
        }
      })
    }

    // Workaround tiling issues by temporarily removing audio source edges
    const audioSourceEdges = cy.edges('[type="audio_source"]').remove()

    // Create and run layout
    const layout = cy.layout({
      ...layoutConfig,
      randomize,
      fixedNodeConstraint,
      tilingCompareBy: (nodeId1: string, nodeId2: string) => {
        if (
          cy?.$id(nodeId1).data('type') === 'audio' &&
          cy?.$id(nodeId2).data('type') === 'audio'
        ) {
          return cy.$id(nodeId1).data('batch_index') - cy.$id(nodeId2).data('batch_index')
        }
        return 0
      }
    } as any)
    layout.run()

    // Restore removed elements
    audioSourceEdges.restore()
  }

  function updateGraph(): void {
    if (!cy || !graphData) return

    try {
      // Store current positions
      const positions: { [key: string]: cytoscape.Position } = {}
      cy.nodes().forEach((node: NodeSingular) => {
        positions[node.id()] = node.position()
      })

      // Clear existing elements
      cy.elements().remove()

      // Add new elements
      if (graphData.elements) {
        cy.add(graphData.elements)

        // Restore positions for existing nodes, but only in 'batch' mode
        if (viewMode === 'batch') {
          cy.nodes().forEach((node: NodeSingular) => {
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
        cy.nodes('[type="batch"]').forEach((node: NodeSingular) => {
          node.data('isExpanded', true)
        })
      }
    } catch (error) {
      console.error('Error updating graph:', error)
    }
  }

  // React to data changes
  $effect(() => {
    if (isInitialized && graphData) {
      updateGraph()
    }
  })

  export function tidyView(): void {
    applyLayout()
  }

  export function fitView(): void {
    cy?.fit()
  }

  export function centerView(): void {
    cy?.center()
  }
</script>

<div class="graph-wrapper">
  <div bind:this={graphContainer} class="graph-container"></div>
  <GraphControls ontidy={() => tidyView()} onfit={() => fitView()} oncenter={() => centerView()} />
  <GraphLegend />
</div>

<style>
  .graph-wrapper {
    position: relative;
    width: 100%;
    flex-grow: 1;
    min-height: 0;
    background: linear-gradient(
      135deg,
      var(--color-background-dark) 0%,
      var(--color-background-medium) 100%
    );
  }

  .graph-container {
    width: 100%;
    height: 100%;
    cursor: grab;
  }

  .graph-container:active {
    cursor: grabbing;
  }
</style>
