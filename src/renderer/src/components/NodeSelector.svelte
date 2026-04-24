<script lang="ts">
  import { onDestroy } from 'svelte'
  import type { ModelData, AudioData } from '../utils/forms'
  import { selectionStore, cyInstanceStore } from '../utils/stores'
  import type { NodeSingular } from 'cytoscape'

  type NodeData = ModelData | AudioData

  let {
    label,
    filter,
    node = $bindable(),
    id,
    onNodeSelect
  } = $props<{
    label?: string
    filter: Record<string, string | number | boolean>
    node: NodeData | string | null
    id: string
    onNodeSelect?: (node: NodeData) => void
  }>()

  let resolvedNode: NodeData | null = $derived.by(() => {
    if (!node) return null
    if (typeof node !== 'string') return node as NodeData
    if (!$cyInstanceStore) return null
    return ($cyInstanceStore.$id(node).data() as NodeData) ?? null
  })

  $effect(() => {
    const cy = $cyInstanceStore
    const target = resolvedNode

    // Explicitly typed reference for the cleanup function
    let boundElement: NodeSingular | null = null

    if (cy && target?.id) {
      const element = cy.$id(target.id)

      // Cytoscape returns a collection; .isNode() ensures it's a node
      if (element.length > 0 && element.isNode()) {
        element.addClass('bound')
        boundElement = element as NodeSingular
      }
    }

    // This return path is now guaranteed
    return () => {
      if (boundElement) {
        boundElement.removeClass('bound')
      }
    }
  })

  $effect(() => {
    // Auto-resolve node if it was passed as a string ID (e.g., from form initialization)
    if (typeof node === 'string' && $cyInstanceStore) {
      selectNodeById(node)
    }
  })

  let isSelecting = false

  function selectNodeFromGraph(): void {
    isSelecting = true
    selectionStore.startSelection(filter, resolvedNode?.id ?? null, (selected) => {
      isSelecting = false
      if (selected) {
        const selectedNode = selected as NodeData
        node = selectedNode
        if (onNodeSelect) onNodeSelect(selectedNode)
      }
    })
  }

  onDestroy(() => {
    if (isSelecting) selectionStore.cancelSelection()
  })

  export function selectNodeById(targetId: string): void {
    const cy = $cyInstanceStore
    if (!cy) return

    const element = cy.$id(targetId)
    if (element.length > 0 && element.isNode()) {
      node = element.data() as NodeData
    } else {
      console.warn(`NodeSelector: Node ${targetId} not found in active graph.`)
    }
  }

  function focusNode(): void {
    console.log('[NodeSelector] focusNode triggered. resolvedNode:', resolvedNode)
    const cy = $cyInstanceStore
    if (!cy) {
      console.warn('[NodeSelector] Cytoscape instance not found')
      return
    }
    if (!resolvedNode?.id) {
      console.warn('[NodeSelector] resolvedNode or resolvedNode.id is missing')
      return
    }
    const ele = cy.$id(resolvedNode.id)
    console.log(
      '[NodeSelector] Found element in graph:',
      ele && ele.length > 0 ? ele.data() : 'None found'
    )
    if (ele && ele.length > 0) {
      cy.animate({ center: { eles: ele }, zoom: 1.5 }, { duration: 400 })
    } else {
      console.warn(
        `[NodeSelector] Element with ID ${resolvedNode.id} not found in the active graph.`
      )
    }
  }
</script>

<div class="form-field">
  {#if label}
    <label for={id}>{label}</label>
  {/if}
  <div class="model-selector-wrapper">
    <input
      {id}
      type="text"
      readonly
      value={resolvedNode ? resolvedNode.alias || resolvedNode.name : 'None selected'}
      onclick={focusNode}
      class:has-node={!!resolvedNode}
      placeholder="Select a node..."
    />
    <button type="button" onclick={selectNodeFromGraph}> Select </button>
  </div>
</div>

<style>
  .form-field {
    margin-bottom: 1rem;
  }
  .form-field label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
  }
  .model-selector-wrapper {
    display: flex;
    gap: 0.5rem;
  }
  .model-selector-wrapper input {
    flex-grow: 1;
    padding: 0.5rem;
    border-radius: 0.375rem;
    border: 1px solid var(--color-border-glass-1);
    background-color: var(--color-background-glass-2);
    color: var(--color-overlay-text);
    min-width: 0;
    cursor: default;
  }
  .model-selector-wrapper input.has-node {
    cursor: pointer;
    border-color: var(--color-primary-faded, #666);
  }
  .model-selector-wrapper input.has-node:hover {
    border-color: var(--color-primary, #999);
  }
  .model-selector-wrapper button {
    flex-shrink: 0;
    padding: 0.5rem 1rem;
    cursor: pointer;
  }
</style>
