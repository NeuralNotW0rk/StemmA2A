<!-- src/renderer/src/components/ParameterGraph.svelte -->
<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  import { onMount, onDestroy } from 'svelte'
  import cytoscape, { type EventObject } from 'cytoscape'
  import fcose from 'cytoscape-fcose'
  import cxtmenu from 'cytoscape-cxtmenu'
  import expandCollapse from 'cytoscape-expand-collapse'
  import graphStyle from './Style'
  import layoutConfig from './Layout'
  import expandCollapseOptions from './Options'
  import { cyInstanceStore, selectionStore } from '../../utils/stores'

  interface Props {
    graphData?: { elements: cytoscape.ElementDefinition[] } | null
    viewMode?: 'batch' | 'cluster'
    onmodelSelect?: (data: any) => void
    onaudioSelect?: (data: any) => void
    onaudioNodeSelectForGeneration?: (data: any, useContext?: boolean) => void
    onexport?: (data: { names: string[] }) => void
    onrescanSource?: (name: string) => void
    onnodeSelect?: (data: any) => void
    onedgeSelect?: (data: any) => void
    onnodeRemove?: (data: any) => void
  }

  let {
    graphData = null,
    onmodelSelect,
    onaudioSelect,
    onaudioNodeSelectForGeneration,
    onrescanSource,
    onnodeSelect,
    onedgeSelect,
    onnodeRemove
  }: Props = $props()

  let graphContainer: HTMLElement | undefined = $state()
  let cy: cytoscape.Core | null = $state(null)
  let isInitialized = $state(false)

  let isSelecting = $state(false)
  let selectionType: 'model' | 'audio' | null = $state(null)
  let selectionBoundNodeId: string | null = $state(null)
  selectionStore.subscribe((store) => {
    isSelecting = store.isSelecting
    selectionType = store.selectionType
    selectionBoundNodeId = store.boundNodeId
  })

  $effect(() => {
    if (cy) {
      // Depend on graphData so this runs after nodes are added.
      if (!graphData) return

      cy.nodes().removeClass('highlighted').removeClass('dimmed').removeClass('bound')

      if (isSelecting) {
        cy.nodes(`[type = "${selectionType}"]`).addClass('highlighted')
        cy.nodes(`[type != "${selectionType}"]`).addClass('dimmed')

        if (selectionBoundNodeId) {
          cy.getElementById(selectionBoundNodeId).removeClass('dimmed').addClass('bound')
        }
      }
    }
  })

  onMount(() => {
    initializeGraph()
  })

  onDestroy(() => {
    cyInstanceStore.set(null) // Clean up
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

    cyInstanceStore.set(cy) // Share the instance globally

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
      selector: 'core',
      commands: [
        {
          content: 'Tidy',
          select: tidyView
        },
        {
          content: 'Fit',
          select: fitView
        }
      ]
    })

    cy.cxtmenu({
      selector: 'node[type="model"]',
      commands: [
        {
          content: 'Generate Audio',
          select: (ele: any) => onmodelSelect?.(ele.data())
        },
        {
          content: 'Remove',
          select: (ele: any) => onnodeRemove?.(ele.data())
        }
      ]
    })

    cy.cxtmenu({
      selector: 'node[type="audio"]',
      commands: (ele: any) => {
        const commands = [
          {
            content: 'Audio to Audio',
            select: () => onaudioNodeSelectForGeneration?.(ele.data(), false)
          },
          {
            content: 'Remove',
            select: () => onnodeRemove?.(ele.data())
          }
        ]

        if (ele.data().context) {
          commands.unshift({
            content: 'Replicate',
            select: () => onaudioNodeSelectForGeneration?.(ele.data(), true)
          })
        }

        return commands
      }
    })

    cy.cxtmenu({
      selector: 'node[type="batch"]',
      commands: [
        {
          content: 'Expand/Collapse',
          select: (ele: any) => {
            const api = cy?.expandCollapse('get')
            if (ele.hasClass('cy-expand-collapse-collapsed')) {
              api.expand(ele)
            } else {
              api.collapse(ele)
            }
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
        },
        {
          content: 'Remove',
          select: (ele: any) => onnodeRemove?.(ele.data())
        }
      ]
    })
  }

  function setupEventListeners(): void {
    if (!cy) return

    cy.on('tap', (evt: EventObject) => {
      if (evt.target === cy) {
        onnodeSelect?.(null)
        onedgeSelect?.(null)
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

    cy.on('tap', 'edge', (evt: EventObject) => {
      const edge = evt.target
      onedgeSelect?.(edge.data())
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

    // Use `as any` to bypass incorrect type definition for `elements`
    const elements = graphData.elements as any
    let newElements: cytoscape.ElementDefinition[] = []

    if (typeof elements === 'undefined') {
      return // No change, elements not provided.
    }

    if (Array.isArray(elements)) {
      newElements = elements
    } else if (elements && Array.isArray(elements.nodes) && Array.isArray(elements.edges)) {
      newElements = elements.nodes.concat(elements.edges)
    } else {
      console.error('graphData.elements has an unexpected format:', elements)
      // Clear the graph if the format is unknown or explicitly null/empty
      cy.elements().remove()
      return
    }

    const newElementIds = new Set(newElements.map((el) => el.data.id))
    const elementsToRemove = cy.elements().filter((ele) => !newElementIds.has(ele.id()))

    if (elementsToRemove.length > 0) {
      cy.remove(elementsToRemove)
    }

    if (newElements.length > 0) {
      cy.add(newElements)
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
