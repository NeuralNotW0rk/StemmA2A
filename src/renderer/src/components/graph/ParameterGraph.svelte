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
    onstartBatching
  }: Props = $props()

  let graphContainer: HTMLElement | undefined = $state()
  let cy: cytoscape.Core | null = $state(null)
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
  }

  function applyLayout(randomize = false): void {
    if (!cy) return
    const layout = cy.layout({ ...layoutConfig, randomize } as any)
    layout.run()
  }

  function updateGraph(use_proxy_edges = false): void {
    if (!cy || !graphData) return

    const elements = graphData.elements as any
    if (typeof elements === 'undefined') return

    let newElements: cytoscape.ElementDefinition[] = []
    if (Array.isArray(elements)) {
      newElements = elements
    } else if (elements?.nodes && elements?.edges) {
      newElements = [...elements.nodes, ...elements.edges]
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

    cy.batch(() => {
      cy.elements().remove()

      let processedElements = newElements
      if (use_proxy_edges) {
        processedElements = newElements.map((ele) => {
          if (ele.group === 'edges' || (ele as any).data.source) {
            const edgeData = (ele as any).data

            // Find the actual nodes in your newElements list to check for parents
            const sourceNode = newElements.find((n) => n.data.id === edgeData.source)
            const targetNode = newElements.find((n) => n.data.id === edgeData.target)

            return {
              ...ele,
              data: {
                ...edgeData,
                // Redirect to parent if it exists, otherwise keep original
                source: sourceNode?.data.parent || edgeData.source,
                target: targetNode?.data.parent || edgeData.target
              }
            }
          }
          return ele
        })
      }
      const addedElements = cy.add(processedElements)
      let hasNewNodes = false

      addedElements.nodes().forEach((node) => {
        const isChildless = node.children().length === 0
        const oldPos = savedPositions[node.id()]

        if (isChildless) {
          if (oldPos) {
            node.position(oldPos)
          } else {
            hasNewNodes = true // Truly new leaf node
          }
        }
      })

      // 4. Only layout if we have nowhere to put new nodes or it's the first run
      if (wasEmpty || hasNewNodes) {
        applyLayout()
      }
    })
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
