<!-- src/renderer/src/components/ParameterGraph.svelte -->
<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  import { onMount, onDestroy } from 'svelte'
  import { SvelteSet } from 'svelte/reactivity'
  import cytoscape, { type EventObject, type Singular } from 'cytoscape'
  import fcose from 'cytoscape-fcose'
  import cxtmenu from 'cytoscape-cxtmenu'
  import graphStyle from './Style'
  import layoutConfig from './Layout'
  import { cyInstanceStore, selectionStore } from '../../utils/stores'
  import { BATCHING_CONFIG } from '../../utils/app-config'

  interface Props {
    graphData?: { elements: cytoscape.ElementDefinition[] } | null
    viewMode?: 'batch' | 'cluster'
    showSpringEdges?: boolean
    showDetailedLabels?: boolean
    onmodelSelect?: (data: any) => void
    onaudioSelect?: (data: any) => void
    onaudioNodeSelectForGeneration?: (data: any, useContext?: boolean) => void
    onaudioNodeSelectForInversion?: (data: any) => void
    onexport?: (data: { names: string[] }) => void
    onimportLattice?: (data: any) => void
    onlatticeSelectForGeneration?: (data: any) => void
    onrescanSource?: (name: string) => void
    onnodeSelect?: (data: any) => void
    onlatentSelectForGeneration?: (data: any) => void
    onedgeSelect?: (data: any) => void
    onElementRemove?: (data: any) => void
    onstartBatching?: (data: any) => void
    onsavePositions?: (positions: Record<string, { x: number; y: number }>) => void | Promise<void>
    onexpandPath?: (id: string) => void
    ontoggleFavorite?: (data: any, isFavorite: boolean) => void
    onupdateElement?: (id: string, attributes: Record<string, any>) => void
    onchangeBatchMembership?: (
      nodeId: string,
      oldBatchId: string | null,
      oldMembers: string[],
      newBatchId: string | null,
      newMembers: string[]
    ) => void | Promise<void>
  }

  let {
    graphData = null,
    showSpringEdges = false,
    showDetailedLabels = true,
    onmodelSelect,
    onaudioSelect,
    onaudioNodeSelectForGeneration,
    onaudioNodeSelectForInversion,
    onimportLattice,
    onlatticeSelectForGeneration,
    onrescanSource,
    onexport,
    onlatentSelectForGeneration,
    onnodeSelect,
    onedgeSelect,
    onElementRemove,
    onstartBatching,
    onsavePositions,
    onexpandPath,
    ontoggleFavorite,
    onupdateElement,
    onchangeBatchMembership
  }: Props = $props()

  let graphContainer: HTMLElement | undefined = $state()
  let cy: cytoscape.Core | null = null
  let isInitialized = $state(false)

  let isSelecting = $state(false)
  let selectionFilter: Record<string, any> | null = $state(null)
  let selectionBoundNodeId: string | null = $state(null)
  selectionStore.subscribe((store) => {
    isSelecting = store.isSelecting
    selectionFilter = store.filter
    selectionBoundNodeId = store.boundNodeId
  })

  function matchesFilter(node: cytoscape.NodeSingular, filter: Record<string, any>): boolean {
    const data = node.data()
    for (const [key, expectedValue] of Object.entries(filter)) {
      // Special evaluation: Exclude specified nodes
      if (key === '_exclude_node_ids') {
        if (Array.isArray(expectedValue)) {
          const strExpected = expectedValue.map(String)
          if (strExpected.includes(String(node.id()))) {
            return false
          }
        }
        continue
      }

      // Special evaluation: Ensure the node has exactly the same incoming structural dependencies
      if (key === '_incoming_node_ids') {
        const incomingIds = node.data('_incoming_node_ids') || []
        if (!Array.isArray(expectedValue) || incomingIds.length !== expectedValue.length)
          return false
        const expectedSorted = [...expectedValue].sort()
        const actualSorted = [...incomingIds].sort()
        for (let i = 0; i < expectedSorted.length; i++) {
          if (expectedSorted[i] !== actualSorted[i]) return false
        }
        continue
      }

      // Standard evaluation: Check nested node data properties
      const keys = key.split('.')
      let actualValue = data
      for (const k of keys) {
        if (actualValue == null) break
        actualValue = actualValue[k]
      }

      if (expectedValue == null) {
        if (actualValue != null && (!Array.isArray(actualValue) || actualValue.length > 0)) {
          return false
        }
      } else if (Array.isArray(expectedValue)) {
        if (!Array.isArray(actualValue) || actualValue.length !== expectedValue.length) return false
        const expectedSorted = [...expectedValue].sort()
        const actualSorted = [...actualValue].sort()
        for (let i = 0; i < expectedSorted.length; i++) {
          if (expectedSorted[i] !== actualSorted[i]) return false
        }
      } else {
        if (actualValue !== expectedValue) return false
      }
    }
    return true
  }

  function isBatchCompatible(batch: cytoscape.NodeSingular, node: cytoscape.NodeSingular): boolean {
    const filter: Record<string, any> = batch.data('member_type') ? { type: batch.data('member_type') } : {}

    // Extract the structural dependencies from the batch's existing members to match BatchingView's logic
    const memberIds = batch.data('member_ids') || []
    if (memberIds.length > 0) {
      const firstMember = cy!.getElementById(memberIds[0])
      if (firstMember && firstMember.length > 0) {
        const incomingIds = firstMember.data('_incoming_node_ids')
        if (incomingIds) filter._incoming_node_ids = incomingIds

        const ctx = firstMember.data('context')
        BATCHING_CONFIG.strictContextKeys.forEach((key) => {
          filter[`context.${key}`] = ctx?.[key] ?? null
        })
      }
    }
    return matchesFilter(node, filter)
  }

  $effect(() => {
    if (isInitialized && cy && graphData) {
      cy.batch(() => {
        // We do NOT remove 'bound' here. NodeSelectors naturally manage their own 'bound' class!
        cy.elements().removeClass('highlighted dimmed bound-active bound-other')

        if (isSelecting && selectionFilter) {
          // 1. Dim everything by default
          cy.elements().addClass('dimmed')

          // 2. Identify valid candidates based on the filter
          const highlightedNodes = cy.nodes().filter((node) => matchesFilter(node, selectionFilter!))
          const ancestorNodes = highlightedNodes.ancestors()

          // 3. Un-dim and highlight valid candidates, and keep their structural ancestors visible
          highlightedNodes.removeClass('dimmed').addClass('highlighted')
          ancestorNodes.removeClass('dimmed')

          // 4. Force the active bound node to show its indicator and un-dim it
          if (selectionBoundNodeId) {
            cy.getElementById(String(selectionBoundNodeId))
              .removeClass('highlighted dimmed')
              .addClass('bound-active')
          }
        }
      })
    }
  })

  $effect(() => {
    if (isInitialized && cy) {
      if (showDetailedLabels) {
        cy.nodes().addClass('detailed')
      } else {
        cy.nodes().removeClass('detailed')
      }
    }
  })

  // Dynamically toggle spring edge visibility whenever the state changes
  $effect(() => {
    if (isInitialized && cy) {
      if (showSpringEdges) {
        cy.edges('edge[type="spring"]').addClass('visible')
      } else {
        cy.edges('edge[type="spring"]').removeClass('visible')
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
          const selectedEles = cy!.$(':selected')
          if (selectedEles.length > 1 && selectedEles.contains(ele)) {
            onElementRemove?.(selectedEles.map(e => e.data()))
          } else {
            onElementRemove?.(ele.data())
          }
        }
      }
    ]

    const nodeCommands = (ele: Singular): Command[] => [
      // Inherits from elementCommands
      ...elementCommands(ele)
    ]

    const modelNodeCommands = (ele: Singular): Command[] => [
      // Inherits from nodeCommands
      {
        content: 'Generate Audio',
        select: () => onmodelSelect?.(ele.data())
      },
      {
        content: 'Import Lattice',
        select: () => onimportLattice?.(ele.data())
      },
      ...nodeCommands(ele)
    ]

    const latticeNodeCommands = (ele: Singular): Command[] => [
      {
        content: 'Generate',
        select: () => onlatticeSelectForGeneration?.(ele.data())
      },
      ...nodeCommands(ele)
    ]

    const latentNodeCommands = (ele: Singular): Command[] => [
      {
        content: 'Generate Audio',
        select: () => onlatentSelectForGeneration?.(ele.data())
      },
      ...nodeCommands(ele)
    ]

    const audioNodeCommands = (ele: Singular): Command[] => {
      const specificCommands: Command[] = [
        {
          content: ele.data('favorite') ? 'Unfavorite' : 'Favorite',
          select: () => {
            const newState = !ele.data('favorite')
            ele.data('favorite', newState)
            ontoggleFavorite?.(ele.data(), newState)
          }
        }
      ]
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
      specificCommands.push({
        content: 'Invert',
        select: () => onaudioNodeSelectForInversion?.(ele.data())
      })

      // Only allow batch creation for independent artifacts
      if (!ele.data('parent')) {
        specificCommands.push({
          content: 'Create Batch',
          select: () => onstartBatching?.(ele.data())
        })
      }
      
      specificCommands.push({
        content: 'Export',
        select: () => onexport?.({ names: [ele.data('name')] })
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

    const pathNodeCommands = (ele: Singular): Command[] => [
      {
        content: 'Expand',
        select: () => onexpandPath?.(ele.id())
      },
      ...nodeCommands(ele)
    ]

    const batchNodeCommands = (ele: Singular): Command[] => [
      {
        content: 'Update Batch',
        select: () => onstartBatching?.(ele.data())
      },
      {
        content: 'Export Batch',
        select: () => {
          const memberIds = ele.data('member_ids') || []
          const names = memberIds.map((id: string) => cy!.getElementById(id).data('name')).filter(Boolean)
          if (names.length > 0) onexport?.({ names })
        }
      },
      ...nodeCommands(ele)
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
          case 'lattice':
            return latticeNodeCommands(ele)
          case 'audio':
            return audioNodeCommands(ele)
          case 'external':
            return externalNodeCommands(ele)
          case 'local_path':
            return pathNodeCommands(ele)
          case 'batch':
            return batchNodeCommands(ele)
          case 'latent':
            return latentNodeCommands(ele)
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
        console.log('[ParameterGraph] Node tapped during selection:', nodeData.id)

        if (!selectionFilter || matchesFilter(node, selectionFilter)) {
          console.log('[ParameterGraph] Valid node clicked. Resolving selection:', nodeData)
          // Pass a clean clone of the data to avoid Svelte 5 proxy / Cytoscape internal conflicts
          selectionStore.resolveSelection({ ...nodeData })
        } else {
          console.warn(`[ParameterGraph] Invalid selection. Node does not match filter.`)
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

    cy.on('dblclick', 'node, edge', (evt: EventObject) => {
      customFit(evt.target, getDynamicPadding())
    })

    cy.on('dblclick', (evt: EventObject) => {
      if (evt.target === cy) {
        fitView()
      }
    })

    let ghostNode: cytoscape.NodeCollection | null = null
    let sourceNode: Singular | null = null

    cy.on('mousedown', 'node', (evt: EventObject) => {
      const node = evt.target
      const oe = evt.originalEvent as MouseEvent

      if (node.children().length === 0 && oe && (oe.ctrlKey || oe.metaKey)) {
        sourceNode = node
        ghostNode = cy!.add({
          group: 'nodes',
          data: { id: `ghost-node-${Date.now()}` },
          position: { ...node.position() }
        })

        ghostNode.style({
          'background-color': node.style('background-color'),
          'shape': node.style('shape'),
          'width': node.style('width'),
          'height': node.style('height'),
          'label': node.style('label'),
          'opacity': 0.5,
          'border-width': node.style('border-width'),
          'border-color': node.style('border-color'),
          'text-valign': node.style('text-valign'),
          'text-halign': node.style('text-halign'),
          'color': node.style('color'),
          'z-index': 9999,
          'events': 'no' // Do not capture events on the ghost node
        })

        cy!.nodes('[type="batch"]').forEach((batch) => {
          if (batch.id() === sourceNode!.id()) return
          if (isBatchCompatible(batch, sourceNode!)) {
            batch.addClass('compatible-drop-target')
          }
        })
      }
    })

    cy.on('mousemove', (evt: EventObject) => {
      if (ghostNode && sourceNode) {
        ghostNode.position(evt.position)

        const nodePos = ghostNode.position()
        cy!.nodes('.compatible-drop-target').forEach((batch) => {
          const bb = batch.boundingBox()
          if (
            nodePos.x >= bb.x1 &&
            nodePos.x <= bb.x2 &&
            nodePos.y >= bb.y1 &&
            nodePos.y <= bb.y2
          ) {
            batch.addClass('active-drop-target')
          } else {
            batch.removeClass('active-drop-target')
          }
        })
      }
    })

    cy.on('mouseup', async (evt: EventObject) => {
      if (ghostNode && sourceNode) {
        const nodePos = ghostNode.position()
        let targetBatchId: string | null = null
        let invalidDrop = false

        cy!.nodes('[type="batch"]').forEach((batch) => {
          if (batch.id() === sourceNode!.id()) return
          const bb = batch.boundingBox()
          if (
            nodePos.x >= bb.x1 &&
            nodePos.x <= bb.x2 &&
            nodePos.y >= bb.y1 &&
            nodePos.y <= bb.y2
          ) {
            if (batch.hasClass('compatible-drop-target')) {
              targetBatchId = batch.id()
            } else {
              invalidDrop = true
            }
          }
        })

        // Ensure classes are cleared regardless of whether the drop was valid
        cy!.nodes('[type="batch"]').removeClass('compatible-drop-target active-drop-target')

        if (invalidDrop && !targetBatchId) {
          ghostNode.remove()
          ghostNode = null
          sourceNode = null
          return
        }

        const currentParentId = sourceNode.data('parent') || null
        const sourceNodeId = sourceNode.id()

        if (currentParentId !== targetBatchId) {
          const oldMembers = currentParentId
            ? cy!.getElementById(currentParentId).data('member_ids')?.filter((id: string) => id !== sourceNodeId) || []
            : []
          const newMembers = targetBatchId
            ? [...(cy!.getElementById(targetBatchId).data('member_ids') || []), sourceNodeId]
            : []

          // 1. Set the physical position immediately before snapshotting
          sourceNode.position({ ...nodePos })

          // 2. Remove ghost node early so it isn't swept into the graph rebuild
          ghostNode.remove()
          ghostNode = null

          // 3. Temporarily mutate local graphData to accurately recalculate proxy edges
          if (graphData && graphData.elements) {
            const elementsList = Array.isArray(graphData.elements)
              ? graphData.elements
              : [...(graphData.elements.nodes || []), ...(graphData.elements.edges || [])]
            
            const rawNode = elementsList.find((e: any) => e.data.id === sourceNodeId)
            if (rawNode) {
              if (targetBatchId) {
                rawNode.data.parent = targetBatchId
              } else {
                delete rawNode.data.parent
              }
            }
          }

          // 4. Instantly re-aggregate and redraw proxy edges without network delay
          updateGraph()

          // 5. Persist
          await extractAndSavePositions()
          if (onchangeBatchMembership) {
            await onchangeBatchMembership(sourceNodeId, currentParentId, oldMembers, targetBatchId, newMembers)
          }
        } else {
          sourceNode.position({ ...nodePos })
          extractAndSavePositions()
        }

        if (ghostNode) ghostNode.remove()
        ghostNode = null
        sourceNode = null
      }
    })

    cy.on('dragfree', 'node', () => {
      // Always save position so it remains physically exactly where it was dropped
      extractAndSavePositions()
    })
  }

  function getDynamicPadding(): { top: number; bottom: number; left: number; right: number } {
    const screenWidth = typeof window !== 'undefined' ? window.innerWidth : 1200
    // ~320px panel + ~30px for margins/gaps. Limit to 30% of screen to keep graph visible on smaller resolutions.
    const sidePadding = Math.min(350, screenWidth * 0.3)
    return {
      top: 60,
      bottom: 60,
      left: sidePadding,
      right: sidePadding
    }
  }

  function customFit(
    eles: any,
    padding: { top: number; bottom: number; left: number; right: number }
  ): void {
    if (!cy) return
    const targetEles = eles && eles.length > 0 ? eles : cy.elements()
    if (targetEles.length === 0) return

    const bb = targetEles.boundingBox()
    const w = Math.max(1, cy.width() - padding.left - padding.right)
    const h = Math.max(1, cy.height() - padding.top - padding.bottom)

    const zoomX = bb.w > 0 ? w / bb.w : cy.maxZoom()
    const zoomY = bb.h > 0 ? h / bb.h : cy.maxZoom()
    const zoom = Math.min(zoomX, zoomY)

    const clampedZoom = Math.max(cy.minZoom(), Math.min(cy.maxZoom(), zoom))

    const pan = {
      x: padding.left + w / 2 - (bb.x1 + bb.w / 2) * clampedZoom,
      y: padding.top + h / 2 - (bb.y1 + bb.h / 2) * clampedZoom
    }

    cy.animate({
      zoom: clampedZoom,
      pan: pan,
      duration: 250,
      easing: 'ease-out-quad'
    })
  }

  function applyLayout(randomize = false, fit = false): void {
    if (!cy) return

    // Save viewport state before layout to restore it if not fitting
    const currentZoom = cy.zoom()
    const currentPan = { ...cy.pan() }

    const elementsToLayout = cy.elements()

    const layout = elementsToLayout.layout({
      ...layoutConfig,
      randomize,
      fit: false, // Prevent the layout algorithm from doing its own bounding box fitting
      animate: true
    } as any)
    layout.on('layoutstop', () => {
      cy?.nodes().unlock()
      if (!fit) {
        cy?.viewport({ zoom: currentZoom, pan: currentPan })
      } else {
        customFit(cy?.elements(), getDynamicPadding())
      }
      extractAndSavePositions()
    })
    layout.run()
  }

  async function extractAndSavePositions(): Promise<void> {
    if (!cy) return
    const positions: Record<string, { x: number; y: number }> = {}
    cy.nodes().forEach((node) => {
      if (node.children().length === 0) {
        positions[node.id()] = { ...node.position() }
      }
    })
    await onsavePositions?.(positions)
  }

  function updateGraph(use_proxy_edges = true): void {
    if (!cy) return

    if (!graphData || !graphData.elements) {
      cy.elements().remove()
      return
    }

    const elements = graphData.elements as any

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

      // Pre-calculate raw structural dependencies before proxy-edge mutations
      const incomingMap: Record<string, string[]> = {}
      newElements.forEach((ele: any) => {
        if ((ele.group === 'edges' || ele.data.source) && ele.data.type !== 'spring') {
          const t = ele.data.target

          const sourceId = ele.data.source

          if (!incomingMap[t]) incomingMap[t] = []
          if (!incomingMap[t].includes(sourceId)) incomingMap[t].push(sourceId)
        }
      })

      const seenProxyEdges = new SvelteSet<string>()

      let processedElements = newElements.reduce((acc: any[], ele: any) => {
        // Backend saves position inside element 'data' attributes, but Cytoscape expects it at the root
        if (ele.data && ele.data.position) {
          ele.position = { ...ele.data.position }
        }

        if (ele.group === 'nodes' || (!ele.data.source && !ele.data.target)) {
          ele.data._incoming_node_ids = incomingMap[ele.data.id] || []
        }

        if (use_proxy_edges && (ele.group === 'edges' || ele.data.source)) {
          const edgeData = ele.data

          // Ignore spring edges so they always connect to specific nodes, not parent batches
          if (edgeData.type === 'spring') {
            acc.push(ele)
            return acc
          }

          // Find the actual nodes in your newElements list to check for parents
          const targetNode = newElements.find((n) => n.data.id === edgeData.target)
          const target = targetNode?.data.parent || edgeData.target

          // We explicitly DO NOT proxy the source to its parent batch.
          // This ensures that when a single sample is used as init_audio, 
          // the edge visually originates from that specific sample, not the whole batch.
          const source = edgeData.source

          // Prevent alpha-stacking visual bugs by deduplicating edges that share the same endpoints
          const sig = `${source}->${target}:${edgeData.type}`
          if (seenProxyEdges.has(sig)) return acc
          seenProxyEdges.add(sig)

          acc.push({
            ...ele,
            data: {
              ...edgeData,
              // Redirect target to parent if it exists, but keep original source
              source: source,
              target: target
            }
          })
        } else {
          acc.push(ele)
        }
        return acc
      }, [])

      const addedElements = cy.add(processedElements)

      // Re-apply visibility to new edges after graph rebuild
      if (showSpringEdges) {
        addedElements.filter('edge[type="spring"]').addClass('visible')
      }
      if (showDetailedLabels) {
        addedElements.nodes().addClass('detailed')
      }

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
      applyLayout(false, wasEmpty)
    } else {
      cy.nodes().unlock()
      if (wasEmpty) {
        customFit(undefined, getDynamicPadding())
      }
    }
  }

  $effect(() => {
    if (isInitialized) {
      updateGraph()
    }
  })

  export function tidyView(): void {
    applyLayout(false, false)
  }

  export function fitView(): void {
    customFit(undefined, getDynamicPadding())
  }

  export function centerView(): void {
    cy?.center()
  }
</script>

<svelte:window
  onkeydown={(e) => {
    if (e.key === 'Control' || e.key === 'Meta') {
      cy?.boxSelectionEnabled(false)
      cy?.autoungrabify(true)
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      const target = e.target as HTMLElement
      const isInput =
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      if (!isInput) {
        const selectedEles = cy?.$(':selected')
        if (selectedEles && selectedEles.length > 0) {
          onElementRemove?.(selectedEles.map((el) => el.data()))
        }
      }
    }
  }}
  onkeyup={(e) => {
    if (e.key === 'Control' || e.key === 'Meta') {
      cy?.boxSelectionEnabled(true)
      cy?.autoungrabify(false)
    }
  }}
  onblur={() => {
    cy?.boxSelectionEnabled(true)
    cy?.autoungrabify(false)
  }}
/>

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
