<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import type { ModelData, AudioData } from '../utils/forms'
  import { selectionStore, cyInstanceStore } from '../utils/stores'
  import type { NodeSingular } from 'cytoscape'
  import NodeSelector from './NodeSelector.svelte'

  type NodeData = ModelData | AudioData

  let {
    label,
    filter,
    node = $bindable(),
    id,
    allowBatchToggle = true,
    autoActivateSelection = false,
    isNested = false,
    onNodeSelect,
    onBatchToggle
  } = $props<{
    label?: string
    filter: Record<string, string | number | boolean>
    node: NodeData | string | null
    id: string
    allowBatchToggle?: boolean
    autoActivateSelection?: boolean
    isNested?: boolean
    onNodeSelect?: (node: NodeData) => void
    onBatchToggle?: (active: boolean) => void
  }>()

  onMount(() => {
    if (autoActivateSelection) {
      setTimeout(() => {
        selectNodeFromGraph()
      }, 50)
    }
  })

  let resolvedNode: NodeData | null = $derived.by(() => {
    if (!node) return null
    if (typeof node === 'object') return node as NodeData
    if (typeof node === 'string' && !node.includes(',')) {
      if (!$cyInstanceStore) return null
      return ($cyInstanceStore.$id(node).data() as NodeData) ?? null
    }
    return null
  })

  $effect(() => {
    const cy = $cyInstanceStore
    const target = resolvedNode

    let boundElement: NodeSingular | null = null

    if (cy && target?.id) {
      const element = cy.$id(target.id)

      if (element.length > 0 && element.isNode()) {
        element.addClass('bound')
        boundElement = element as NodeSingular
      }
    }

    return () => {
      if (boundElement) {
        boundElement.removeClass('bound')
      }
    }
  })

  $effect(() => {
    // Auto-resolve node if it was passed as a string ID (e.g., from form initialization)
    if (!isBatchMode && node && typeof node === 'string' && !node.includes(',') && $cyInstanceStore) {
      selectNodeById(node)
    }
  })

  let isBatchMode = $state(false)
  const showBatchToggle = $derived(allowBatchToggle)

  let nextBatchItemId = 0
  let batchItems = $state<{ id: number; node: NodeData | string | null; autoActivate?: boolean }[]>([])

  // Watch for changes in isBatchMode to sync node and batchItems
  $effect(() => {
    if (isBatchMode) {
      if (batchItems.length === 0) {
        if (node) {
          let ids: string[] = []
          if (typeof node === 'object' && node !== null && 'id' in node) {
            if (node.type === 'batch' && (node as any).member_ids && $cyInstanceStore) {
              ids = (node as any).member_ids as string[]
            } else {
              ids = [node.id]
            }
          } else {
            const cy = $cyInstanceStore
            const nodeStr = String(node)
            if (cy && cy.$id(nodeStr).data('type') === 'batch') {
              ids = cy.$id(nodeStr).data('member_ids') || []
            } else {
              ids = nodeStr.split(',').map(s => s.trim()).filter(Boolean)
            }
          }
          batchItems = ids.map((id) => ({ id: nextBatchItemId++, node: id, autoActivate: false }))
        } else {
          batchItems = [{ id: nextBatchItemId++, node: null, autoActivate: false }]
        }
      }
    } else {
      batchItems = []
    }
  })

  // Watch for changes in nested batchItems to update the parent node value
  $effect(() => {
    if (isBatchMode && batchItems.length > 0) {
      const ids = batchItems
        .map(item => {
          const n = item.node
          if (!n) return ''
          return typeof n === 'string' ? n : n.id
        })
        .filter(Boolean)
      
      const newStringVal = ids.join(', ')
      if (node !== newStringVal) {
        node = newStringVal
      }
    }
  })

  $effect(() => {
    // Auto-enable batch mode if node is a comma-separated list of IDs
    if (allowBatchToggle && node && typeof node === 'string' && node.includes(',')) {
      isBatchMode = true
      if (onBatchToggle) {
        onBatchToggle(true)
      }
    }
  })

  let isSelecting = false

  function selectNodeFromGraph(): void {
    isSelecting = true
    console.log('[NodeSelector] Starting graph selection. Filter:', JSON.stringify(filter))
    selectionStore.startSelection(filter, resolvedNode?.id ?? null, (selected) => {
      isSelecting = false
      if (selected) {
        const selectedNode = selected as NodeData
        
        // If selecting a Batch/container node in single-selection mode, auto-toggle batch mode and unpack
        if (!isBatchMode && allowBatchToggle && selectedNode.type === 'batch' && (selectedNode as any).member_ids && $cyInstanceStore) {
          isBatchMode = true
          if (onBatchToggle) {
            onBatchToggle(true)
          }
          
          const cy = $cyInstanceStore
          const memberIds = (selectedNode as any).member_ids as string[]
          const memberNodes = memberIds
            .map(mId => cy.$id(mId).data() as NodeData)
            .filter(Boolean)
            
          batchItems = memberNodes.map((member) => ({ id: nextBatchItemId++, node: member, autoActivate: false }))
        } else {
          node = selectedNode
          if (onNodeSelect) onNodeSelect(selectedNode)
        }
      }
    })
  }

  function toggleBatchMode() {
    isBatchMode = !isBatchMode
    if (onBatchToggle) {
      onBatchToggle(isBatchMode)
    }
    
    if (!isBatchMode) {
      // Transitioning back to single selection: take the first non-empty node if available
      const firstNonEmpty = batchItems.find(item => item.node !== null)
      node = firstNonEmpty ? firstNonEmpty.node : null
    }
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

  function handleChildNodeSelect(selectedNode: NodeData, index: number) {
    if (selectedNode.type === 'batch' && (selectedNode as any).member_ids && $cyInstanceStore) {
      const cy = $cyInstanceStore
      const memberIds = (selectedNode as any).member_ids as string[]
      const memberNodes = memberIds
        .map(mId => cy.$id(mId).data() as NodeData)
        .filter(Boolean)
      
      if (memberNodes.length > 0) {
        // Clear the current slot so we don't bind to the batch node itself
        batchItems[index].node = null

        const existingIds = new Set(
          batchItems
            .map(item => {
              const n = item.node
              if (!n) return ''
              return typeof n === 'string' ? n : n.id
            })
            .filter(Boolean)
        )

        let isFirst = true
        for (const member of memberNodes) {
          if (isFirst) {
            batchItems[index].node = member
            isFirst = false
          } else {
            if (!existingIds.has(member.id)) {
              batchItems.push({ id: nextBatchItemId++, node: member, autoActivate: false })
            }
          }
        }
      }
    }
  }
</script>

<div class="form-field" class:nested={isNested}>
  {#if label}
    <label for={id}>{label}</label>
  {/if}
  
  {#if isBatchMode}
    <div class="batch-list-container">
      {#each batchItems as item, index (item.id)}
        <div class="batch-item-row">
          <div class="batch-item-selector">
            <NodeSelector
              filter={filter}
              bind:node={batchItems[index].node}
              allowBatchToggle={false}
              isNested={true}
              autoActivateSelection={item.autoActivate}
              onNodeSelect={(selectedNode) => {
                handleChildNodeSelect(selectedNode, index)
              }}
              id="{id}-batch-{item.id}"
            />
          </div>
          <button
            type="button"
            class="remove-item-btn"
            onclick={() => {
              batchItems = batchItems.filter(i => i.id !== item.id)
              if (batchItems.length === 0) {
                batchItems = [{ id: nextBatchItemId++, node: null, autoActivate: false }]
              }
            }}
            title="Remove from batch"
          >
            ✕
          </button>
        </div>
      {/each}
      <div class="batch-actions-row">
        <button
          type="button"
          class="add-item-btn"
          onclick={() => {
            batchItems.push({ id: nextBatchItemId++, node: null, autoActivate: true })
          }}
        >
          + Add Node
        </button>
        {#if allowBatchToggle}
          <button
            type="button"
            class="action-btn"
            onclick={toggleBatchMode}
            title="Disable batch mode"
          >
            −
          </button>
        {/if}
      </div>
    </div>
  {:else}
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
      {#if showBatchToggle}
        <button
          type="button"
          class="action-btn"
          onclick={toggleBatchMode}
          title="Enable batch mode"
        >
          +
        </button>
      {/if}
    </div>
  {/if}
</div>

<style>
  .form-field {
    margin-bottom: 1rem;
  }
  .form-field.nested {
    margin-bottom: 0;
  }
  .form-field label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
  }
  .model-selector-wrapper {
    display: flex;
    align-items: center;
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
  .action-btn {
    background: none;
    border: 1px solid var(--color-overlay-border-primary, var(--color-border-glass-1));
    color: var(--color-overlay-text);
    cursor: pointer;
    width: 24px;
    height: 24px;
    min-width: 24px;
    min-height: 24px;
    flex: 0 0 24px;
    border-radius: 50%;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    box-sizing: border-box;
    align-self: center;
  }
  .batch-list-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.5rem;
    border: 1px dashed var(--color-border-glass-1);
    border-radius: 0.375rem;
    background-color: rgba(255, 255, 255, 0.02);
  }
  .batch-item-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .batch-item-selector {
    flex-grow: 1;
  }
  .remove-item-btn {
    flex-shrink: 0;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 1px solid var(--color-border-glass-1);
    background: none;
    color: var(--color-text-muted, #aaa);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    padding: 0;
  }
  .remove-item-btn:hover {
    background-color: var(--color-error, #ef4444);
    border-color: var(--color-error, #ef4444);
    color: white;
  }
  .batch-actions-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.25rem;
  }
  .add-item-btn {
    background: none;
    border: 1px dashed var(--color-border-glass-1);
    color: var(--color-overlay-text);
    padding: 0.25rem 0.75rem;
    border-radius: 0.25rem;
    cursor: pointer;
    font-size: 0.8rem;
  }
  .add-item-btn:hover {
    border-color: var(--color-primary);
    color: var(--color-primary);
  }
</style>
