<!-- src/renderer/src/components/ParameterGraph.svelte -->
<script lang="ts">
  /* eslint-disable @typescript-eslint/no-explicit-any */
  import { onMount, onDestroy } from 'svelte'
  import { SvelteSet } from 'svelte/reactivity'
  import cytoscape, { type EventObject, type Singular } from 'cytoscape'
  import fcose from 'cytoscape-fcose'
  import cxtmenu from 'cytoscape-cxtmenu'
  import popper from 'cytoscape-popper'
  import { createPopper } from '@popperjs/core'
  import graphStyle from './Style'
  import layoutConfig from './Layout'
  import NodePropertyPopper from './NodePropertyPopper.svelte'
  import { cyInstanceStore, selectionStore } from '../../utils/stores'
  import { GROUPING_CONFIG } from '../../utils/app-config'

  interface Props {
    graphData?: {
      elements: cytoscape.ElementDefinition[] | cytoscape.ElementsDefinition
    } | null
    viewMode?: 'batch' | 'cluster'
    showSpringEdges?: boolean
    showDetailedLabels?: boolean
    chronologicalConstraint?: boolean
    operations?: any[]
    onaudioSelect?: (data: any) => void
    onexport?: (data: { names: string[] }) => void
    onimportGrating?: (data: any) => void
    onbendModel?: (data: any) => void
    onrescanSource?: (name: string) => void
    onnodeSelect?: (data: any) => void
    onedgeSelect?: (data: any) => void
    onElementRemove?: (data: any) => void
    onstartGrouping?: (data: any) => void
    onsavePositions?: (positions: Record<string, { x: number; y: number }>) => void | Promise<void>
    onexpandPath?: (id: string) => void
    ontoggleFavorite?: (data: any, isFavorite: boolean) => void
    onchangeGroupMembership?: (
      nodeId: string,
      oldGroupId: string | null,
      oldMembers: string[],
      newGroupId: string | null,
      newMembers: string[]
    ) => void | Promise<void>
    onselectOperation?: (op: any, initiatorNode: any, useContext?: boolean) => void
    onrefresh?: () => void | Promise<void>
  }

  let {
    graphData = null,
    showSpringEdges = false,
    showDetailedLabels = true,
    chronologicalConstraint = false,
    operations: operationsProp = [],
    onaudioSelect,
    onimportGrating,
    onbendModel,
    onrescanSource,
    onexport,
    onnodeSelect,
    onedgeSelect,
    onElementRemove,
    onstartGrouping,
    onsavePositions,
    onexpandPath,
    ontoggleFavorite,
    onchangeGroupMembership,
    onselectOperation,
    onrefresh
  }: Props = $props()

  let graphContainer: HTMLElement | undefined = $state()
  let cy: cytoscape.Core | null = null
  let isInitialized = $state(false)

  // --- In-place Audio Operations Popover State ---
  let operationsMenuOpen = $state(false)
  let targetNode: any = $state(null)
  let operations: any[] = $state([])
  let searchQuery = $state('')

  $effect(() => {
    if (operationsProp && operationsProp.length > 0) {
      operations = operationsProp
    }
  })

  let selectedIndex = $state(0)
  let menuElement: HTMLElement | null = $state(null)
  let searchInput: HTMLInputElement | null = $state(null)
  let loadingOperations = $state(false)
  let operationsError: string | null = $state(null)
  let lastCxtTapPosition = { x: 0, y: 0 }

  // --- Inline Node Property Popper State ---
  interface NodePropertyPopperConfig {
    targetNode: cytoscape.NodeSingular
    title: string
    inputType?: 'number' | 'text'
    initialValue?: string | number | null
    placeholder?: string
    submitLabel?: string
    allowClear?: boolean
    clearLabel?: string
    step?: string
    onsubmit: (value: string | number | null) => Promise<void> | void
  }
  let activePropertyPopper = $state<NodePropertyPopperConfig | null>(null)

  const displayType = $derived(
    targetNode
      ? targetNode.data('type') === 'group'
        ? targetNode.data('member_type')
        : targetNode.data('type')
      : null
  )

  const outputType = $derived.by(() => {
    if (!targetNode) return null
    if (targetNode.data('type') === 'group') {
      const memberIds = targetNode.data('member_ids') || []
      if (memberIds.length > 0 && cy) {
        const firstMember = cy.getElementById(memberIds[0])
        if (firstMember && firstMember.length > 0) {
          return firstMember.data('output_type') || null
        }
      }
    }
    return targetNode.data('output_type') || null
  })

  function hasAssociatedModel(node: any): boolean {
    if (!node) return false
    const data = typeof node.data === 'function' ? node.data() : node
    if (!data) return false
    if (data.context?.model_id || data.context?.model || data.base_model_id) {
      return true
    }
    if (
      typeof node.predecessors === 'function' &&
      node.predecessors('node[type = "model"]').length > 0
    ) {
      return true
    }
    if (data.type === 'group' && Array.isArray(data.member_ids) && data.member_ids.length > 0 && cy) {
      const firstMember = cy.getElementById(data.member_ids[0])
      if (firstMember && firstMember.length > 0) {
        return hasAssociatedModel(firstMember)
      }
    }
    return false
  }

  let filteredOperations = $derived.by(() => {
    const initiatorType = targetNode ? targetNode.data('type') : null
    let ops = operations
    if (initiatorType) {
      const filterType = initiatorType === 'group' ? targetNode.data('member_type') : initiatorType
      if (filterType) {
        ops = operations.filter((op) => {
          return (
            !op.initiator_types ||
            op.initiator_types.length === 0 ||
            op.initiator_types.includes(filterType)
          )
        })
      }
    }

    // Filter out 'mutate' if target audio node does not have an associated generative model
    if (targetNode && (initiatorType === 'audio' || targetNode.data('member_type') === 'audio')) {
      if (!hasAssociatedModel(targetNode)) {
        ops = ops.filter((op) => op.name !== 'mutate')
      }
    }

    if (!searchQuery) return ops
    const q = searchQuery.toLowerCase()
    return ops.filter((op) => {
      const display = getOpDisplayProps(op, displayType, outputType)
      return (
        display.name.toLowerCase().includes(q) ||
        (display.description && display.description.toLowerCase().includes(q))
      )
    })
  })

  function getOpDisplayProps(
    op: any,
    initiatorType?: string,
    outputType?: string
  ): { name: string; description?: string } {
    if (op.name === 'generate' && initiatorType === 'model') {
      if (outputType) {
        return {
          name: `Generate ${toTitleCase(outputType)}`,
          description: `Generate ${outputType} from the selected model`
        }
      } else {
        return {
          name: `Generate`,
          description: op.description || `Generate output from the selected model`
        }
      }
    }
    if (initiatorType && op.context_overrides && op.context_overrides[initiatorType]) {
      const override = op.context_overrides[initiatorType]
      return {
        name: override.name || op.name,
        description: override.description || op.description
      }
    }
    return {
      name: op.name,
      description: op.description
    }
  }

  function toTitleCase(str: string): string {
    return str
      .split(/[\s-_]+/)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  async function openOperationsMenu(node: any): Promise<void> {
    targetNode = node
    operationsMenuOpen = true
    searchQuery = ''
    selectedIndex = 0
    loadingOperations = true
    operationsError = null

    try {
      const response = await window.api.getOperations()
      if (response && response.success && Array.isArray(response.operations)) {
        operations = response.operations
      } else {
        throw new Error('Invalid operations response format')
      }
    } catch (err: any) {
      console.error('Failed to get operations:', err)
      operationsError = err.message || String(err)
    } finally {
      loadingOperations = false
      setTimeout(() => {
        if (searchInput) searchInput.focus()
      }, 50)
    }
  }

  function handleKeyDown(e: KeyboardEvent): void {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (filteredOperations.length > 0) {
        selectedIndex = (selectedIndex + 1) % filteredOperations.length
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      if (filteredOperations.length > 0) {
        selectedIndex = (selectedIndex - 1 + filteredOperations.length) % filteredOperations.length
      }
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (filteredOperations[selectedIndex]) {
        selectOp(filteredOperations[selectedIndex])
      }
    } else if (e.key === 'Escape') {
      e.preventDefault()
      closeOperationsMenu()
    }
  }

  function selectOp(op: any): void {
    const nodeData = targetNode ? targetNode.data() : null
    closeOperationsMenu()
    if (onselectOperation && nodeData) {
      onselectOperation(op, nodeData)
    }
  }

  function closeOperationsMenu(): void {
    operationsMenuOpen = false
    targetNode = null
  }

  function handleClickOutside(e: MouseEvent): void {
    if (operationsMenuOpen && menuElement && !menuElement.contains(e.target as Node)) {
      closeOperationsMenu()
    }
  }

  let popperInstance: any = null

  $effect(() => {
    if (operationsMenuOpen && menuElement && targetNode && cy) {
      const ref = targetNode.popperRef()
      popperInstance = createPopper(ref, menuElement, {
        placement: 'right-start',
        strategy: 'fixed',
        modifiers: [
          {
            name: 'offset',
            options: {
              offset: [0, 10]
            }
          }
        ]
      })

      const update = (): void => {
        if (popperInstance) popperInstance.update()
      }

      targetNode.on('position', update)
      cy.on('pan zoom resize', update)

      return () => {
        if (targetNode) targetNode.off('position', update)
        if (cy) cy.off('pan zoom resize', update)
        if (popperInstance) {
          popperInstance.destroy()
          popperInstance = null
        }
      }
    }
    return undefined
  })

  function openScorePopper(ele: Singular): void {
    const node = ele as cytoscape.NodeSingular
    const currentFitness = node.data('fitness')
    activePropertyPopper = {
      targetNode: node,
      title: 'Fitness Score',
      inputType: 'number',
      initialValue: currentFitness !== undefined && currentFitness !== null ? currentFitness : null,
      placeholder: 'Score (e.g. 1.0, 0.85)',
      submitLabel: 'Set Score',
      allowClear: true,
      onsubmit: async (val: string | number | null): Promise<void> => {
        const numVal = val !== null && val !== '' ? Number(val) : null
        node.data('fitness', numVal)
        await window.api.updateElement(node.id(), { fitness: numVal })
        onnodeSelect?.(node.data())
      }
    }
  }

  function openRenamePopper(ele: Singular): void {
    const node = ele as cytoscape.NodeSingular
    const currentName = (node.data('name') || node.data('alias') || node.id()) as string
    activePropertyPopper = {
      targetNode: node,
      title: 'Rename Node',
      inputType: 'text',
      initialValue: currentName,
      placeholder: 'Enter new name...',
      submitLabel: 'Rename',
      allowClear: false,
      onsubmit: async (val: string | number | null): Promise<void> => {
        const newName = val == null ? '' : String(val).trim()
        if (!newName) {
          throw new Error('Name cannot be empty.')
        }
        node.data('name', newName)
        await window.api.updateElement(node.id(), { name: newName })
        onnodeSelect?.(node.data())
      }
    }
  }

  function matchesFilter(node: cytoscape.NodeSingular, filter: Record<string, any>): boolean {
    const data = node.data()

    if (data.type === 'group') {
      const memberIds = data.member_ids || []
      if (memberIds.length === 0) return false
      const firstMember = node.cy().getElementById(memberIds[0])
      if (!firstMember || firstMember.length === 0) return false
      return matchesFilter(firstMember, filter)
    }
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

  function isGroupCompatible(group: cytoscape.NodeSingular, node: cytoscape.NodeSingular): boolean {
    const filter: Record<string, any> = group.data('member_type')
      ? { type: group.data('member_type') }
      : {}

    // Extract the structural dependencies from the group's existing members to match GroupingView's logic
    const memberIds = group.data('member_ids') || []
    if (memberIds.length > 0) {
      const firstMember = cy!.getElementById(memberIds[0])
      if (firstMember && firstMember.length > 0) {
        const incomingIds = firstMember.data('_incoming_node_ids')
        if (incomingIds) filter._incoming_node_ids = incomingIds

        const ctx = firstMember.data('context')
        GROUPING_CONFIG.strictContextKeys.forEach((key) => {
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

        if ($selectionStore.isSelecting) {
          const filter = $selectionStore.filter || {}
          console.log('[ParameterGraph] Selection active. Filter is:', JSON.stringify(filter))
          // 1. Dim everything by default
          cy.elements().addClass('dimmed')

          // 2. Identify valid candidates based on the filter
          const highlightedNodes = cy.nodes().filter((node) => matchesFilter(node, filter))
          const ancestorNodes = highlightedNodes.ancestors()

          // 3. Un-dim and highlight valid candidates, and keep their structural ancestors visible
          highlightedNodes.removeClass('dimmed').addClass('highlighted')
          ancestorNodes.removeClass('dimmed')

          // 4. Force the active bound node to show its indicator and un-dim it
          if ($selectionStore.boundNodeId) {
            cy.getElementById(String($selectionStore.boundNodeId))
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

  // Run layout when chronological constraint changes to show the immediate effect
  let firstEffectRun = true
  $effect(() => {
    void chronologicalConstraint
    if (isInitialized && cy) {
      if (firstEffectRun) {
        firstEffectRun = false
        return
      }
      tidyView()
    }
  })

  onMount(() => {
    initializeGraph()
  })

  let removeKeydownListener: (() => void) | null = null

  onDestroy(() => {
    removeKeydownListener?.()
    cyInstanceStore.set(null) // Clean up
    if (cy) {
      cy.destroy()
    }
  })

  async function initializeGraph(): Promise<void> {
    cytoscape.use(fcose)
    cytoscape.use(cxtmenu)
    cytoscape.use(popper(createPopper))

    try {
      const response = await window.api.getOperations()
      if (response && response.success && Array.isArray(response.operations)) {
        operations = response.operations
      }
    } catch (e) {
      console.error('Failed to load operations for context menus:', e)
    }

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

    async function handleExpressIndividual(individualId: string): Promise<void> {
      try {
        await window.api.expressIndividual({ individual_id: individualId })
        await onrefresh?.()
      } catch (err: any) {
        console.error('Failed to express individual:', err)
      }
    }

    const PINNED_OPERATIONS = new Set(['generate', 'invert'])

    const getPinnedOperations = (ele: Singular): Command[] => {
      if (!operations) return []
      const type = ele.data('type')
      return operations
        .filter(
          (op) =>
            PINNED_OPERATIONS.has(op.name) &&
            op.initiator_types &&
            op.initiator_types.includes(type)
        )
        .map((op) => {
          const display = getOpDisplayProps(op, type)
          return {
            content: toTitleCase(display.name),
            select: () => {
              onselectOperation?.(op, ele.data())
            }
          }
        })
    }

    const getReplicateCommand = (ele: Singular): Command | null => {
      const context = ele.data().context
      if (!context) return null

      let targetOpName = context.operation
      if (!targetOpName) {
        // Fallback for legacy elements
        const type = ele.data('type')
        if (type === 'audio' || type === 'image') {
          targetOpName = 'generate'
        } else if (type === 'latent') {
          targetOpName = 'invert'
        }
      }

      if (!targetOpName) return null

      const targetOp = operations?.find((op) => op.name === targetOpName)
      if (!targetOp) return null

      return {
        content: 'Replicate',
        select: () => {
          if (onselectOperation) {
            onselectOperation(targetOp, ele.data(), true)
          }
        }
      }
    }

    const elementCommands = (ele: Singular): Command[] => [
      {
        content: 'Remove',
        select: () => {
          const selectedEles = cy!.$(':selected')
          if (selectedEles.length > 1 && selectedEles.contains(ele)) {
            onElementRemove?.(selectedEles.map((e) => e.data()))
          } else {
            onElementRemove?.(ele.data())
          }
        }
      }
    ]

    const nodeCommands = (ele: Singular): Command[] => [
      {
        content: 'Rename',
        select: () => openRenamePopper(ele)
      },
      ...elementCommands(ele)
    ]

    const modelNodeCommands = (ele: Singular): Command[] => [
      ...getPinnedOperations(ele),
      {
        content: 'Bend',
        select: () => onbendModel?.(ele.data())
      },
      {
        content: 'Import Grating',
        select: () => onimportGrating?.(ele.data())
      },
      ...nodeCommands(ele)
    ]

    const gratingNodeCommands = (ele: Singular): Command[] => [
      ...getPinnedOperations(ele),
      ...nodeCommands(ele)
    ]

    const individualNodeCommands = (ele: Singular): Command[] => [
      {
        content: 'Express',
        select: () => handleExpressIndividual(ele.id())
      },
      {
        content: 'Score',
        select: () => openScorePopper(ele)
      },
      ...nodeCommands(ele)
    ]

    const latentNodeCommands = (ele: Singular): Command[] => {
      const specificCommands: Command[] = []
      const replicateCmd = getReplicateCommand(ele)
      if (replicateCmd) {
        specificCommands.push(replicateCmd)
      }
      specificCommands.push(...getPinnedOperations(ele))
      return [...specificCommands, ...nodeCommands(ele)]
    }

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

      const replicateCmd = getReplicateCommand(ele)
      if (replicateCmd) {
        specificCommands.push(replicateCmd)
      }

      specificCommands.push({
        content: 'Export',
        select: () => onexport?.({ names: [ele.data('name')] })
      })

      specificCommands.push({
        content: 'Operations...',
        select: () => openOperationsMenu(ele)
      })

      // Inherits from nodeCommands
      return [...specificCommands, ...nodeCommands(ele)]
    }

    const imageNodeCommands = (ele: Singular): Command[] => {
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

      const replicateCmd = getReplicateCommand(ele)
      if (replicateCmd) {
        specificCommands.push(replicateCmd)
      }

      specificCommands.push(...getPinnedOperations(ele))

      specificCommands.push({
        content: 'Operations...',
        select: () => openOperationsMenu(ele)
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

    const groupNodeCommands = (ele: Singular): Command[] => [
      {
        content: 'Update Group',
        select: () => onstartGrouping?.(ele.data())
      },
      {
        content: 'Export Group',
        select: () => {
          const memberIds = ele.data('member_ids') || []
          const names = memberIds
            .map((id: string) => cy!.getElementById(id).data('name'))
            .filter(Boolean)
          if (names.length > 0) onexport?.({ names })
        }
      },
      {
        content: 'Operations...',
        select: () => openOperationsMenu(ele)
      },
      ...nodeCommands(ele)
    ]

    // --- Menu Instantiation ---

    cy.cxtmenu({
      selector: 'core',
      commands: [
        { content: 'Tidy', select: tidyView },
        { content: 'Fit', select: fitView },
        {
          content: 'Create Group',
          select: async () => {
            try {
              const response = await window.api.groupElements([])
              if (response && response.success) {
                const newGroup = response.collection
                await onsavePositions?.({
                  [newGroup.id]: { x: lastCxtTapPosition.x, y: lastCxtTapPosition.y }
                })
                await onrefresh?.()
              }
            } catch (err: any) {
              console.error('Failed to create empty group:', err)
            }
          }
        }
      ]
    })

    cy.cxtmenu({
      selector: 'node',
      commands: (ele: Singular): Command[] => {
        switch (ele.data('type')) {
          case 'model':
            return modelNodeCommands(ele)
          case 'grating':
            return gratingNodeCommands(ele)
          case 'audio':
            return audioNodeCommands(ele)
          case 'image':
            return imageNodeCommands(ele)
          case 'external':
            return externalNodeCommands(ele)
          case 'local_path':
            return pathNodeCommands(ele)
          case 'group':
            return groupNodeCommands(ele)
          case 'individual':
            return individualNodeCommands(ele)
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

    cy.on('cxttapstart', (evt: EventObject) => {
      if (evt.position) {
        lastCxtTapPosition = { ...evt.position }
        console.log('[ParameterGraph] Position updated to:', lastCxtTapPosition)
      }
    })

    cy.on('tap', (evt: EventObject) => {
      if (evt.target === cy) {
        onnodeSelect?.(null)
        onedgeSelect?.(null)
      }
    })

    cy.on('tap', 'node', (evt: EventObject) => {
      const node = evt.target
      const nodeData = node.data()

      if ($selectionStore.isSelecting) {
        console.log('[ParameterGraph] Node tapped during selection:', nodeData.id)

        if (!$selectionStore.filter || matchesFilter(node, $selectionStore.filter)) {
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

    let ghostNode: cytoscape.NodeSingular | null = null
    let sourceNode: cytoscape.NodeSingular | null = null
    let trackedMousedownNode: cytoscape.NodeSingular | null = null
    let trackedInitialPositions: Record<string, { x: number; y: number }> = {}
    let lastRenderedMousePos: { x: number; y: number } | null = null
    let isMouseDown = false

    function cleanupGhost(): void {
      if (ghostNode) {
        ghostNode.remove()
        ghostNode = null
      }
      sourceNode = null
      trackedMousedownNode = null
      trackedInitialPositions = {}
      lastRenderedMousePos = null
      isMouseDown = false
      cy?.nodes('[type="group"]').removeClass('compatible-drop-target active-drop-target')
    }

    function startGhostDrag(
      node: cytoscape.NodeSingular,
      initialPos: { x: number; y: number }
    ): void {
      if (ghostNode) return

      // Restore original node (and any children) back to their initial positions before normal drag moved them
      if (trackedInitialPositions[node.id()]) {
        node.position({ ...trackedInitialPositions[node.id()] })
      }
      node.children().forEach((child) => {
        if (trackedInitialPositions[child.id()]) {
          child.position({ ...trackedInitialPositions[child.id()] })
        }
      })

      sourceNode = node
      ghostNode = cy!.add({
        group: 'nodes',
        data: { id: `ghost-node-${Date.now()}` },
        position: { ...initialPos }
      }) as unknown as cytoscape.NodeSingular

      ghostNode.style({
        'background-color': node.style('background-color'),
        shape: node.style('shape'),
        'corner-radius': node.style('corner-radius'),
        width: node.style('width'),
        height: node.style('height'),
        label: node.style('label'),
        opacity: 0.5,
        'border-width': node.style('border-width'),
        'border-color': node.style('border-color'),
        'border-style': node.style('border-style'),
        'text-valign': node.style('text-valign'),
        'text-halign': node.style('text-halign'),
        color: node.style('color'),
        'z-index': 9999,
        events: 'no' // Do not capture events on the ghost node
      })

      cy!.nodes('[type="group"]').forEach((group) => {
        if (group.id() === sourceNode!.id()) return
        if (isGroupCompatible(group as unknown as cytoscape.NodeSingular, sourceNode!)) {
          group.addClass('compatible-drop-target')
        }
      })
    }

    const handleKeyDown = (e: KeyboardEvent): void => {
      if (
        (e.key === 'Control' || e.key === 'Meta' || e.ctrlKey || e.metaKey) &&
        isMouseDown &&
        !ghostNode &&
        trackedMousedownNode
      ) {
        startGhostDrag(
          trackedMousedownNode,
          lastRenderedMousePos || trackedMousedownNode.position()
        )
      }
    }

    const handleWindowMouseUp = (): void => {
      if (isMouseDown || ghostNode) {
        handleMouseUp()
      }
    }

    const handleWindowBlur = (): void => {
      cleanupGhost()
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('mouseup', handleWindowMouseUp)
    window.addEventListener('blur', handleWindowBlur)

    removeKeydownListener = () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('mouseup', handleWindowMouseUp)
      window.removeEventListener('blur', handleWindowBlur)
    }

    cy.on('mousedown', 'node', (evt: EventObject) => {
      const rawTarget = evt.target
      const oe = evt.originalEvent as MouseEvent

      // If dragging an exemplar inside an individual compound node, resolve target to the individual itself
      const parent = rawTarget.parent()
      const isInsideIndividual = parent && parent.length > 0 && parent.data('type') === 'individual'
      const node = isInsideIndividual ? parent : rawTarget

      const nodeType = node.data('type')
      const nodeParent = node.parent()
      const nodeParentType = nodeParent && nodeParent.length > 0 ? nodeParent.data('type') : null

      // Block ghost-dragging for members of directory nodes, directories, and groups
      const isGhostDraggable =
        nodeParentType !== 'directory' && nodeType !== 'directory' && nodeType !== 'group'

      const isEligible = node.children().length === 0 || nodeType === 'individual'

      if (isEligible && isGhostDraggable) {
        isMouseDown = true
        trackedMousedownNode = node as unknown as cytoscape.NodeSingular
        trackedInitialPositions = { [node.id()]: { ...node.position() } }
        node.children().forEach((child: any) => {
          trackedInitialPositions[child.id()] = { ...child.position() }
        })
        lastRenderedMousePos = { ...evt.position }

        if (oe && (oe.ctrlKey || oe.metaKey)) {
          startGhostDrag(trackedMousedownNode, evt.position)
        }
      }
    })

    cy.on('mousemove', (evt: EventObject) => {
      lastRenderedMousePos = { ...evt.position }
      const oe = evt.originalEvent as MouseEvent

      // If mouse buttons are not pressed, clean up any phantom drag state immediately
      if (oe && oe.buttons === 0) {
        if (ghostNode || isMouseDown) {
          cleanupGhost()
        }
        return
      }

      if (isMouseDown && !ghostNode && trackedMousedownNode && oe && (oe.ctrlKey || oe.metaKey)) {
        startGhostDrag(trackedMousedownNode, evt.position)
      }

      if (ghostNode && sourceNode) {
        ghostNode.position(evt.position)

        const nodePos = ghostNode.position()
        cy!.nodes('.compatible-drop-target').forEach((group) => {
          const bb = group.boundingBox()
          if (
            nodePos.x >= bb.x1 &&
            nodePos.x <= bb.x2 &&
            nodePos.y >= bb.y1 &&
            nodePos.y <= bb.y2
          ) {
            group.addClass('active-drop-target')
          } else {
            group.removeClass('active-drop-target')
          }
        })
      }
    })

    async function handleMouseUp(): Promise<void> {
      isMouseDown = false

      if (ghostNode && sourceNode) {
        const nodePos = ghostNode.position()
        let targetGroupId: string | null = null
        let invalidDrop = false

        cy!.nodes('[type="group"]').forEach((group) => {
          if (group.id() === sourceNode!.id()) return
          const bb = group.boundingBox()
          if (
            nodePos.x >= bb.x1 &&
            nodePos.x <= bb.x2 &&
            nodePos.y >= bb.y1 &&
            nodePos.y <= bb.y2
          ) {
            if (group.hasClass('compatible-drop-target')) {
              targetGroupId = group.id()
            } else {
              invalidDrop = true
            }
          }
        })

        if (invalidDrop && !targetGroupId) {
          cleanupGhost()
          return
        }

        const currentParentId = sourceNode.data('parent') || null
        const sourceNodeId = sourceNode.id()
        const currentSourceNode = sourceNode

        // Remove ghost node early before any graph updates
        if (ghostNode) {
          ghostNode.remove()
          ghostNode = null
        }
        sourceNode = null

        if (currentParentId !== targetGroupId) {
          const oldMembers = currentParentId
            ? cy!
                .getElementById(currentParentId)
                .data('member_ids')
                ?.filter((id: string) => id !== sourceNodeId) || []
            : []
          const newMembers = targetGroupId
            ? [...(cy!.getElementById(targetGroupId).data('member_ids') || []), sourceNodeId]
            : []

          // 1. Set physical position: if sourceNode has children (e.g. compound individual), shift all its children by delta (dx, dy)
          if (currentSourceNode.children().length > 0) {
            const currentPos = currentSourceNode.position()
            const dx = nodePos.x - currentPos.x
            const dy = nodePos.y - currentPos.y
            currentSourceNode.children().forEach((child) => {
              const cp = child.position()
              child.position({ x: cp.x + dx, y: cp.y + dy })
            })
          } else {
            currentSourceNode.position({ ...nodePos })
          }

          // 2. Temporarily mutate local graphData to accurately recalculate proxy edges
          if (graphData && graphData.elements) {
            const elementsList = Array.isArray(graphData.elements)
              ? graphData.elements
              : [...(graphData.elements.nodes || []), ...(graphData.elements.edges || [])]

            const rawNode = elementsList.find((e: any) => e.data.id === sourceNodeId)
            if (rawNode) {
              if (targetGroupId) {
                rawNode.data.parent = targetGroupId
              } else {
                delete rawNode.data.parent
              }
            }
          }

          // 3. Instantly re-aggregate and redraw proxy edges without network delay
          updateGraph()

          // 4. Persist
          await extractAndSavePositions()
          if (onchangeGroupMembership) {
            await onchangeGroupMembership(
              sourceNodeId,
              currentParentId,
              oldMembers,
              targetGroupId,
              newMembers
            )
          }
        } else {
          if (currentSourceNode.children().length > 0) {
            const currentPos = currentSourceNode.position()
            const dx = nodePos.x - currentPos.x
            const dy = nodePos.y - currentPos.y
            currentSourceNode.children().forEach((child) => {
              const cp = child.position()
              child.position({ x: cp.x + dx, y: cp.y + dy })
            })
          } else {
            currentSourceNode.position({ ...nodePos })
          }
          extractAndSavePositions()
        }
      }

      cleanupGhost()
    }

    cy.on('mouseup', handleMouseUp)

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

  function isGeneratedArtifact(node: cytoscape.NodeSingular): boolean {
    const type = node.data('type')
    if (type !== 'audio' && type !== 'latent') {
      return false
    }
    const parentId = node.data('parent')
    if (parentId) {
      const parentNode = cy!.getElementById(parentId)
      if (parentNode && parentNode.length > 0) {
        const parentType = parentNode.data('type')
        if (
          parentType === 'local_path' ||
          parentType === 'directory' ||
          parentType === 'external'
        ) {
          return false
        }
      }
    }
    return true
  }

  function buildChronologicalConstraints(): { left: string; right: string; gap: number }[] {
    if (!cy) return []
    const leafNodes = cy.nodes().filter((node) => node.children().length === 0)
    const sortedNodes = (leafNodes.toArray() as cytoscape.NodeSingular[])
      .filter((node) => node.data('created') !== undefined && isGeneratedArtifact(node))
      .sort((a, b) => {
        const timeA = a.data('created') || 0
        const timeB = b.data('created') || 0
        if (timeA !== timeB) return timeA - timeB
        return a.id().localeCompare(b.id())
      })

    const constraints: { left: string; right: string; gap: number }[] = []
    for (let i = 0; i < sortedNodes.length - 1; i++) {
      constraints.push({
        left: sortedNodes[i].id(),
        right: sortedNodes[i + 1].id(),
        gap: 80
      })
    }
    return constraints
  }

  function applyLayout(randomize = false, fit = false, useFixedConstraints = true): void {
    if (!cy) return

    // Save viewport state before layout to restore it if not fitting
    const currentZoom = cy.zoom()
    const currentPan = { ...cy.pan() }

    const elementsToLayout = cy.elements()

    // Dynamically build fixed node constraints for all nodes that have positions
    const fixedConstraints: { nodeId: string; position: { x: number; y: number } }[] = []
    if (useFixedConstraints) {
      cy.nodes().forEach((node) => {
        const isChildless = node.children().length === 0
        const pos = node.position()
        if (
          isChildless &&
          node.locked() &&
          typeof pos.x === 'number' &&
          typeof pos.y === 'number' &&
          !isNaN(pos.x) &&
          !isNaN(pos.y)
        ) {
          fixedConstraints.push({
            nodeId: node.id(),
            position: { x: pos.x, y: pos.y }
          })
        }
      })
    }

    const layout = elementsToLayout.layout({
      ...layoutConfig,
      randomize,
      fit: false, // Prevent the layout algorithm from doing its own bounding box fitting
      animate: true,
      fixedNodeConstraint: fixedConstraints.length > 0 ? fixedConstraints : undefined,
      relativePlacementConstraint: chronologicalConstraint
        ? buildChronologicalConstraints()
        : undefined
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

    // Filter out baseline nodes and edges connected to them to keep the visual graph clean
    const baselineNodeIds = new Set<string>()
    newElements = newElements.filter((ele: any): boolean => {
      if (ele.group === 'nodes' && ele.data.type === 'grating' && ele.data.context?.is_baseline) {
        baselineNodeIds.add(ele.data.id)
        return false
      }
      return true
    })
    newElements = newElements.filter((ele: any): boolean => {
      if (ele.group === 'edges' || ele.data.source) {
        if (baselineNodeIds.has(ele.data.source) || baselineNodeIds.has(ele.data.target)) {
          return false
        }
      }
      return true
    })

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
        if ((ele.group === 'edges' || ele.data?.source) && ele.data?.type !== 'spring') {
          const t = ele.data.target
          const sourceId = ele.data.source

          if (t && sourceId) {
            if (!incomingMap[t]) incomingMap[t] = []
            if (!incomingMap[t].includes(sourceId)) incomingMap[t].push(sourceId)
          }
        }
      })

      // Pre-calculate which incoming node IDs are shared by ALL members of each parent/group
      const allNodes = newElements.filter(
        (ele: any) => ele.group === 'nodes' || (!ele.data?.source && !ele.data?.target)
      )
      const parentMembersMap: Record<string, any[]> = {}
      allNodes.forEach((n: any) => {
        const parentId = n.data?.parent
        if (parentId) {
          if (!parentMembersMap[parentId]) parentMembersMap[parentId] = []
          parentMembersMap[parentId].push(n)
        }
      })

      const parentSharedIncomingMap: Record<string, SvelteSet<string>> = {}
      Object.entries(parentMembersMap).forEach(([parentId, members]) => {
        if (members.length === 0) return
        const firstMemberIncoming = incomingMap[members[0].data?.id] || []
        const shared = firstMemberIncoming.filter((sourceId) =>
          members.every((m) => incomingMap[m.data?.id]?.includes(sourceId))
        )
        parentSharedIncomingMap[parentId] = new SvelteSet(shared)
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

        const isEdge = ele.group === 'edges' || Boolean(ele.data?.source && ele.data?.target)

        if (isEdge) {
          const edgeData = ele.data
          const source = edgeData.source
          const targetNode = newElements.find((n) => n.data.id === edgeData.target)
          const sourceNode = newElements.find((n) => n.data.id === edgeData.source)

          // Skip edges between a compound parent and its direct child (composition is already visual)
          if (
            targetNode?.data.parent === source ||
            sourceNode?.data.parent === edgeData.target ||
            source === edgeData.target
          ) {
            return acc
          }

          if (use_proxy_edges) {
            // Ignore spring edges so they always connect to specific nodes, not parent batches
            if (edgeData.type === 'spring') {
              acc.push(ele)
              return acc
            }

            // We explicitly DO NOT proxy the source to its parent batch.
            // This ensures that when a single sample is used as init_audio,
            // the edge visually originates from that specific sample, not the whole batch.
            let target = edgeData.target
            const parentId = targetNode?.data?.parent

            // If ALL members of the parent group share this incoming source,
            // aggregate and redirect the target to the group itself.
            if (parentId && parentId !== source && parentSharedIncomingMap[parentId]?.has(source)) {
              target = parentId
            }

            if (source === target) return acc

            // Prevent alpha-stacking visual bugs by deduplicating edges that share the same endpoints
            const sig = `${source}->${target}:${edgeData.type}`
            if (seenProxyEdges.has(sig)) return acc
            seenProxyEdges.add(sig)

            acc.push({
              ...ele,
              data: {
                ...edgeData,
                // Redirect target to parent if shared by all members, but keep original source
                source: source,
                target: target
              }
            })
            return acc
          } else {
            acc.push(ele)
            return acc
          }
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
    applyLayout(false, false, false)
  }

  export function fitView(): void {
    customFit(undefined, getDynamicPadding())
  }

  export function centerView(): void {
    cy?.center()
  }
</script>

<svelte:window
  onmousedown={handleClickOutside}
  onkeydown={(e) => {
    if (e.key === 'Control' || e.key === 'Meta') {
      cy?.boxSelectionEnabled(false)
      cy?.autoungrabify(true)
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      const target = e.target as HTMLElement
      const isInput =
        target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable
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

<div class="graph-wrapper" class:selecting={$selectionStore.isSelecting}>
  <div bind:this={graphContainer} class="graph-container"></div>

  {#if operationsMenuOpen}
    <div bind:this={menuElement} class="operations-command-palette">
      <div class="search-header">
        <svg
          class="search-icon"
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        <input
          bind:this={searchInput}
          type="text"
          bind:value={searchQuery}
          onkeydown={handleKeyDown}
          placeholder="Search audio operations..."
        />
      </div>

      <div class="palette-content">
        {#if loadingOperations}
          <div class="status-message">
            <div class="spinner" style="margin-bottom: 0.5rem;"></div>
            Loading operations...
          </div>
        {:else if operationsError}
          <div class="status-message error">{operationsError}</div>
        {:else if filteredOperations.length === 0}
          <div class="status-message">No operations found</div>
        {:else}
          <ul class="operations-list">
            {#each filteredOperations as op, index (op.name)}
              {@const display = getOpDisplayProps(op, displayType, outputType)}
              <li class="operation-item" class:active={index === selectedIndex}>
                <button
                  type="button"
                  onclick={() => selectOp(op)}
                  onmouseenter={() => (selectedIndex = index)}
                >
                  <div class="op-title-row">
                    <span class="op-name">{display.name.toUpperCase()}</span>
                    <span
                      class="op-badge"
                      class:dsp={op.category === 'dsp' || (!op.category && op.execution_mode === 'sync')}
                      class:evolution={op.category === 'evolution'}
                      class:generative={op.category === 'generative' || (!op.category && op.execution_mode === 'async')}
                    >
                      {op.category || op.execution_mode || 'op'}
                    </span>
                  </div>
                  {#if display.description}
                    <div class="op-desc">{display.description}</div>
                  {/if}
                </button>
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    </div>
  {/if}

  {#if activePropertyPopper}
    <NodePropertyPopper
      targetNode={activePropertyPopper.targetNode}
      title={activePropertyPopper.title}
      inputType={activePropertyPopper.inputType}
      initialValue={activePropertyPopper.initialValue}
      placeholder={activePropertyPopper.placeholder}
      submitLabel={activePropertyPopper.submitLabel}
      allowClear={activePropertyPopper.allowClear}
      clearLabel={activePropertyPopper.clearLabel}
      step={activePropertyPopper.step}
      onsubmit={activePropertyPopper.onsubmit}
      onclose={() => (activePropertyPopper = null)}
    />
  {/if}
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

  /* --- Command Palette Custom Styles --- */
  .operations-command-palette {
    position: fixed;
    width: 280px;
    background: var(--color-background-glass-2, rgba(20, 20, 25, 0.85));
    backdrop-filter: blur(12px);
    border: 1px solid var(--color-overlay-border-primary, rgba(255, 255, 255, 0.15));
    border-radius: 8px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    z-index: 10000;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    max-height: 320px;
    font-family: inherit;
  }

  .search-header {
    display: flex;
    align-items: center;
    padding: 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.03);
    gap: 0.5rem;
  }

  .search-icon {
    color: var(--color-text-muted, #888);
    opacity: 0.7;
    flex-shrink: 0;
  }

  .search-header input {
    background: none;
    border: none;
    outline: none;
    color: var(--color-overlay-text, #fff);
    font-size: 0.9rem;
    width: 100%;
    padding: 0;
    margin: 0;
  }

  .palette-content {
    overflow-y: auto;
    flex-grow: 1;
  }

  .status-message {
    padding: 1.5rem 1rem;
    text-align: center;
    color: var(--color-text-muted, #888);
    font-size: 0.85rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  }

  .status-message.error {
    color: var(--color-error, #ef4444);
  }

  .operations-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .operation-item {
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  }

  .operation-item:last-child {
    border-bottom: none;
  }

  .operation-item button {
    display: block;
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    border-radius: 0;
    padding: 0.75rem 1rem;
    color: var(--color-overlay-text, #fff);
    cursor: pointer;
    transition: background 0.15s;
    min-height: auto;
  }

  .operation-item.active button {
    background: var(--color-primary-t-30, rgba(235, 94, 40, 0.2));
  }

  .op-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
  }

  .op-name {
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
  }

  .op-badge {
    font-size: 0.65rem;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.1);
    color: var(--color-text-muted, #aaa);
    text-transform: uppercase;
    font-weight: 500;
  }

  .op-badge.dsp {
    background: rgba(148, 163, 184, 0.15);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.25);
  }

  .op-badge.evolution {
    background: rgba(168, 85, 247, 0.2);
    color: #d8b4fe;
    border: 1px solid rgba(168, 85, 247, 0.35);
  }

  .op-badge.generative {
    background: rgba(59, 130, 246, 0.2);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.35);
  }

  .op-desc {
    font-size: 0.75rem;
    color: var(--color-text-muted, #888);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
