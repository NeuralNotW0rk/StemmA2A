<!-- src/renderer/src/components/ParameterGraph.svelte -->
<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  import { onMount, onDestroy } from 'svelte'
  import cytoscape, { type EventObject, type Singular } from 'cytoscape'
  import fcose from 'cytoscape-fcose'
  import cxtmenu from 'cytoscape-cxtmenu'
  import graphStyle from './Style'
  import layoutConfig from './Layout'
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
    onElementRemove?: (data: any) => void
    onstartBatching?: (data: any) => void
    onsavePositions?: (positions: Record<string, { x: number; y: number }>) => void
  }

  let {
    graphData = null,
    onmodelSelect,
    onaudioSelect,
    onaudioNodeSelectForGeneration,
    onrescanSource,
    onnodeSelect,
    onedgeSelect,
    onElementRemove,
    onstartBatching,
    onsavePositions
  }: Props = $props()

  let graphContainer: HTMLElement | undefined = $state()
  let cy: cytoscape.Core | null = null
  let isInitialized = $state(false)

  let isSelecting = $state(false)
  let selectionFilter: Record<string, string | number | boolean> | null = $state(null)
  let selectionBoundNodeId: string | null = $state(null)
  selectionStore.subscribe((store) => {
    isSelecting = store.isSelecting
    selectionFilter = store.filter
    selectionBoundNodeId = store.boundNodeId
  })

  function buildSelector(filter: Record<string, string | number | boolean>): string {
    return Object.entries(filter)
      .map(([key, value]) => `[${key} = "${value}"]`)
      .join('')
  }

  $effect(() => {
    if (cy) {
      // Depend on graphData so this runs after nodes are added.
      if (!graphData) return

      cy.nodes().removeClass('highlighted').removeClass('dimmed')

      if (isSelecting && selectionFilter) {
        const selector = buildSelector(selectionFilter)

        if (selector) {
          const highlightedNodes = cy.nodes(selector)
          const ancestorNodes = highlightedNodes.ancestors()

          highlightedNodes.addClass('highlighted')
          cy.nodes().not(highlightedNodes).not(ancestorNodes).addClass('dimmed')
        }

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

    cy = cytoscape({
      container: graphContainer,
      style: graphStyle,
      layout: { name: 'grid' },
      wheelSensitivity: 0.2,
      maxZoom: 3,
      minZoom: 0.1
    })

    cyInstanceStore.set(cy) // Share the instance globally

    setupContextMenus()
    setupEventListeners()
    isInitialized = true

    if (graphData) {
      updateGraph()
    }
  }

  function setupContextMenus(): void {
    if (!cy) return

    type Command = { content: string; select: () => void }

    // --- Command Definitions (Inheritance-style) ---

    const elementCommands = (ele: Singular): Command[] => [
      {
        content: 'Remove',
        select: () => {
          onElementRemove?.(ele.data())
        }
      }
    ]

    const nodeCommands = (ele: Singular): Command[] => [
      // Inherits from elementCommands
      ...elementCommands(ele),
      {
        content: 'Group',
        select: () => onstartBatching?.(ele.data())
      }
    ]

    const modelNodeCommands = (ele: Singular): Command[] => [
      // Inherits from nodeCommands
      {
        content: 'Generate Audio',
        select: () => onmodelSelect?.(ele.data())
      },
      ...nodeCommands(ele)
    ]

    const audioNodeCommands = (ele: Singular): Command[] => {
      const specificCommands: Command[] = []
      if (ele.data().context) {
        specificCommands.push({
          content: 'Replicate',
          select: () => onaudioNodeSelectForGeneration?.(ele.data(), true)
        })
      }
      specificCommands.push({
        content: 'Audio to Audio',
        select: () => onaudioNodeSelectForGeneration?.(ele.data(), false)
      })

      // Inherits from nodeCommands
      return [...specificCommands, ...nodeCommands(ele)]
    }

    const externalNodeCommands = (ele: Singular): Command[] => [
      // Inherits from elementCommands (cannot be grouped)
      {
        content: 'Rescan',
        select: () => onrescanSource?.(ele.data().name)
      },
      ...elementCommands(ele)
    ]

    // --- Menu Instantiation ---

    cy.cxtmenu({
      selector: 'core',
      commands: [
        { content: 'Tidy', select: tidyView },
        { content: 'Fit', select: fitView }
      ]
    })

    cy.cxtmenu({
      selector: 'node',
      commands: (ele: Singular): Command[] => {
        switch (ele.data('type')) {
          case 'model':
            return modelNodeCommands(ele)
          case 'audio':
            return audioNodeCommands(ele)
          case 'external':
            return externalNodeCommands(ele)
          default:
            return nodeCommands(ele) // Fallback for any other node type
        }
      }
    })

    cy.cxtmenu({
      selector: 'edge',
      commands: elementCommands
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
        if (
          selectionFilter &&
          Object.entries(selectionFilter).every(([key, value]) => node.data(key) === value)
        ) {
          selectionStore.resolveSelection(nodeData)
        } else {
          // Maybe provide some feedback for wrong selection
          console.log(`Select a node that matches the filter.`)
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

    cy.on('dragfree', 'node', extractAndSavePositions)
  }

  function applyLayout(randomize = false): void {
    if (!cy) return
    const layout = cy.layout({ ...layoutConfig, randomize } as any)
    layout.on('layoutstop', () => {
      cy?.nodes().unlock()
      extractAndSavePositions()
    })
    layout.run()
  }

  function extractAndSavePositions(): void {
    if (!cy) return
    const positions: Record<string, { x: number; y: number }> = {}
    cy.nodes().forEach((node) => {
      if (node.children().length === 0) {
        positions[node.id()] = { ...node.position() }
      }
    })
    onsavePositions?.(positions)
  }

  function updateGraph(use_proxy_edges = true): void {
    if (!cy || !graphData) return

    const elements = graphData.elements as any
    if (typeof elements === 'undefined') return

    // Deep clone to prevent Cytoscape from mutating Svelte proxy state and causing infinite loops
    const clonedElements = JSON.parse(JSON.stringify(elements))

    let newElements: cytoscape.ElementDefinition[] = []
    if (Array.isArray(clonedElements)) {
      newElements = clonedElements
    } else if (clonedElements?.nodes && clonedElements?.edges) {
      newElements = [...clonedElements.nodes, ...clonedElements.edges]
    } else {
      cy.elements().remove()
      return
    }

    // 1. Snapshot the CURRENT visual state
    const savedPositions: Record<string, { x: number; y: number }> = {}
    cy.nodes().forEach((node) => {
      // Only store positions for leaf nodes to avoid compound drift
      if (node.children().length === 0) {
        savedPositions[node.id()] = { ...node.position() }
      }
    })

    const wasEmpty = Object.keys(savedPositions).length === 0
    let requiresLayout = false

    cy.batch(() => {
      cy.elements().remove()

      let processedElements = newElements.map((ele: any) => {
        // Backend saves position inside element 'data' attributes, but Cytoscape expects it at the root
        if (ele.data && ele.data.position) {
          ele.position = { ...ele.data.position }
        }

        if (use_proxy_edges && (ele.group === 'edges' || ele.data.source)) {
          const edgeData = ele.data

          // Ignore spring edges so they always connect to specific nodes, not parent batches
          if (edgeData.type === 'spring') {
            return ele
          }

          // Find the actual nodes in your newElements list to check for parents
          const targetNode = newElements.find((n) => n.data.id === edgeData.target)

          return {
            ...ele,
            data: {
              ...edgeData,
              // Redirect target to parent if it exists, but keep original source
              source: edgeData.source,
              target: targetNode?.data.parent || edgeData.target
            }
          }
        }
        return ele
      })

      const addedElements = cy.add(processedElements)

      addedElements.nodes().forEach((node) => {
        const isChildless = node.children().length === 0
        const oldPos = savedPositions[node.id()]
        const backendPos = node.data('position')

        if (isChildless) {
          if (oldPos) {
            node.position(oldPos)
            node.lock()
          } else if (
            backendPos &&
            typeof backendPos.x === 'number' &&
            typeof backendPos.y === 'number'
          ) {
            node.position(backendPos)
            node.lock()
          } else {
            requiresLayout = true // Truly new leaf node without position
          }
        }
      })
    })

    // 4. Only layout if we have new nodes missing positions
    if (requiresLayout) {
      applyLayout()
    } else {
      cy.nodes().unlock()
      if (wasEmpty) {
        cy.fit(undefined, 50)
      }
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
