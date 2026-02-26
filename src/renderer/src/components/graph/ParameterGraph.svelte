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
  import { selectionStore } from '../../utils/stores'

  interface Props {
    graphData?: { elements: cytoscape.ElementDefinition[] } | null
    viewMode?: 'batch' | 'cluster'
    onmodelSelect?: (data: any) => void
    onaudioSelect?: (data: any) => void
    onaudioNodeSelectForGeneration?: (data: any) => void
    onexport?: (data: { names: string[] }) => void
    onrescanSource?: (name: string) => void
    onnodeSelect?: (data: any) => void
  }

  let {
    graphData = null,
    viewMode = 'batch',
    onmodelSelect,
    onaudioSelect,
    onaudioNodeSelectForGeneration,
    onexport,
    onrescanSource,
    onnodeSelect
  }: Props = $props()

  let graphContainer: HTMLElement | undefined = $state()
  let cy: cytoscape.Core | null = $state(null)
  let isInitialized = $state(false)
  let isSelecting = $state(false)
  let selectionType: 'model' | 'audio' | null = $state(null)

  selectionStore.subscribe((store) => {
    isSelecting = store.isSelecting
    selectionType = store.selectionType
    if (cy) {
      if (store.isSelecting) {
        cy.nodes().removeClass('highlighted').removeClass('dimmed') // Reset first
        cy.nodes(`[type = "${store.selectionType}"]`).addClass('highlighted')
        cy.nodes(`[type != "${store.selectionType}"]`).addClass('dimmed')
      } else {
        cy.nodes().removeClass('highlighted').removeClass('dimmed')
      }
    }
  })

  onMount(() => {
    initializeGraph()
  })

  onDestroy(() => {
    if (cy) {
      cy.destroy()
    }
  })

  function initializeGraph(): void {
    cytoscape.use(fcose)
    cytoscape.use(cxtmenu)
    cytoscape.use(expandCollapse)

    cy = cytoscape({
      container: graphContainer,
      style: graphStyle,
      layout: { name: 'grid' },
      wheelSensitivity: 0.2,
      maxZoom: 3,
      minZoom: 0.1
    })

    cy.expandCollapse({ ...expandCollapseOptions, layoutBy: layoutConfig })

    setupContextMenus()
    setupEventListeners()
    isInitialized = true

    if (graphData) {
      updateGraph()
    }
  }

  function setupContextMenus(): void {
    if (!cy) return

    cy.cxtmenu({
      selector: 'node[type="model"]',
      commands: [
        {
          content: 'Generate Audio',
          select: (ele: any) => onmodelSelect?.(ele.data())
        }
      ]
    })

    cy.cxtmenu({
      selector: 'node[type="audio"]',
      commands: [
        {
          content: 'Play/Select',
          select: (ele: any) => onaudioSelect?.(ele.data())
        },
        {
          content: 'Create Variation',
          select: (ele: any) => onaudioNodeSelectForGeneration?.(ele.data())
        },
        {
          content: 'Export',
          select: (ele: any) => onexport?.({ names: [ele.data().name] })
        }
      ]
    })

    cy.cxtmenu({
      selector: 'node[type="batch"]',
      commands: [
        {
          content: 'Expand/Collapse',
          select: (ele: any) => {
            cy?.expandCollapse('get').collapse(ele).run()
          }
        }
      ]
    })

    cy.cxtmenu({
      selector: 'node[type="external"]',
      commands: [
        {
          content: 'Rescan',
          select: (ele: any) => onrescanSource?.(ele.data().name)
        }
      ]
    })
  }

  function setupEventListeners(): void {
    if (!cy) return

    cy.on('tap', (evt: EventObject) => {
      if (evt.target === cy) {
        onnodeSelect?.(null)
      }
    })

    cy.on('tap', 'node', (evt: EventObject) => {
      const node = evt.target
      const nodeData = node.data()

      if (isSelecting) {
        if (node.data('type') === selectionType) {
          selectionStore.resolveSelection(nodeData)
        } else {
          // Maybe provide some feedback for wrong selection
          console.log(`Select a ${selectionType} node.`)
        }
        return // Prevent default action
      }

      onnodeSelect?.(nodeData)

      if (nodeData.type === 'audio') {
        onaudioSelect?.(nodeData)
      }
    })

    cy.on('dblclick', () => cy?.fit())
  }

  function applyLayout(randomize = false): void {
    if (!cy) return
    const layout = cy.layout({ ...layoutConfig, randomize } as any)
    layout.run()
  }

  function updateGraph(): void {
    if (!cy || !graphData) return
    cy.elements().remove()
    if (graphData.elements) {
      cy.add(graphData.elements)
      applyLayout()
    }
  }

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

<div class="graph-wrapper" class:selecting={isSelecting}>
  <div bind:this={graphContainer} class="graph-container"></div>
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

  .graph-wrapper.selecting .graph-container {
    cursor: crosshair;
  }
</style>
