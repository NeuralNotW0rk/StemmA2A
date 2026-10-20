<!-- src/renderer/src/components/NodeSelectorList.svelte -->
<script lang="ts">
  import { onDestroy, type Snippet } from 'svelte'
  import NodeSelector from './NodeSelector.svelte'
  import { selectionStore, cyInstanceStore } from '../utils/stores'
  import type { NodeData, GroupData } from '../utils/forms'
  import type { NodeFilter, NodeListItem } from '../utils/types'
  import { parseSequence, numericSequenceError } from '../utils/sequence'

  let {
    title = 'Items',
    addButtonText = 'Add',
    filter = {},
    items = $bindable([]),
    idPrefix = 'list-item',
    showStrengths = false,
    strengthStep = 0.1,
    defaultStrength = 1.0,
    onAdd,
    itemExtra
  } = $props<{
    title?: string
    addButtonText?: string
    filter?: NodeFilter
    items: NodeListItem[]
    idPrefix?: string
    minItems?: number
    showStrengths?: boolean
    strengthStep?: number
    defaultStrength?: number
    onAdd?: () => void
    itemExtra?: Snippet<[NodeListItem, number]>
  }>()

  let nextId = $state(0)
  let isSelecting = false

  // Automatically sync internal ID counter to prevent collisions with pre-populated lists
  $effect(() => {
    if (items.length > 0) {
      const numericIds = items.map((i) => Number(i.id)).filter((id) => !isNaN(id))
      if (numericIds.length > 0) {
        const maxId = Math.max(...numericIds)
        if (maxId >= nextId) {
          nextId = maxId + 1
        }
      }
    }
  })

  function handleAdd(): void {
    if (onAdd) {
      onAdd()
    } else {
      const newItem: NodeListItem = { id: nextId++, node: null, strength: defaultStrength }
      items.push(newItem)

      isSelecting = true
      selectionStore.startSelection(filter, null, (selected) => {
        isSelecting = false
        if (!selected) {
          items = items.filter((item) => item.id !== newItem.id)
          return
        }

        const selectedNode = selected as NodeData
        const cy = $cyInstanceStore

        // If a group node was selected, unpack its member nodes into individual top-level items
        if (selectedNode.type === 'group' && (selectedNode as unknown as GroupData).member_ids && cy) {
          const memberIds = (selectedNode as unknown as GroupData).member_ids
          const memberNodes = memberIds
            .map((mId) => cy.$id(mId).data() as NodeData)
            .filter(Boolean)
            .filter((m) => !filter?.type || m.type === filter.type)

          // Remove the placeholder item
          items = items.filter((item) => item.id !== newItem.id)

          if (memberNodes.length > 0) {
            for (const member of memberNodes) {
              items.push({ id: nextId++, node: member, strength: defaultStrength })
            }
          }
        } else {
          const item = items.find((i) => i.id === newItem.id)
          if (item) {
            item.node = selectedNode
          }
        }
      })
    }
  }

  function handleChildNodeSelect(selectedNode: NodeData, index: number): void {
    const cy = $cyInstanceStore
    if (selectedNode.type === 'group' && (selectedNode as unknown as GroupData).member_ids && cy) {
      const memberIds = (selectedNode as unknown as GroupData).member_ids
      const memberNodes = memberIds
        .map((mId) => cy.$id(mId).data() as NodeData)
        .filter(Boolean)
        .filter((m) => !filter?.type || m.type === filter.type)

      if (memberNodes.length > 0) {
        items[index].node = memberNodes[0]
        for (let i = 1; i < memberNodes.length; i++) {
          items.push({ id: nextId++, node: memberNodes[i], strength: defaultStrength })
        }
      } else {
        items[index].node = null
      }
    }
  }

  function toggleStrengthBatch(item: NodeListItem): void {
    if (item.strengthBatch) {
      // Leaving sequence mode: keep the first value of the sequence
      let first: unknown
      try {
        first = parseSequence(String(item.strength ?? ''))[0]
      } catch {
        first = undefined
      }
      item.strength = typeof first === 'number' && isFinite(first) ? first : defaultStrength
    } else {
      item.strength = String(item.strength ?? defaultStrength)
    }
    item.strengthBatch = !item.strengthBatch
  }

  function handleRemove(id: number | string): void {
    items = items.filter((item) => item.id !== id)
  }

  onDestroy(() => {
    if (isSelecting) selectionStore.cancelSelection()
  })
</script>

<div class="node-selector-list">
  <div class="header-row">
    {#if title}
      <span class="list-title">{title}</span>
    {:else}
      <div></div>
    {/if}
    <button type="button" onclick={handleAdd} class="small-btn">{addButtonText}</button>
  </div>
  <div class="members-list">
    {#each items as item, index (item.id)}
      <div class="list-item-container">
        <div class="member-row">
          <div class="member-selector">
            <NodeSelector
              {filter}
              bind:node={items[index].node}
              allowBatchToggle={false}
              id={`${idPrefix}-${item.id}`}
              onNodeSelect={(selectedNode: NodeData): void => {
                handleChildNodeSelect(selectedNode, index)
              }}
            />
          </div>
          <button
            type="button"
            class="remove-button"
            onclick={() => handleRemove(item.id)}
            title="Remove item"
            aria-label="Remove item"
          >
            ✕
          </button>
        </div>
        {#if showStrengths && items[index].node}
          <div class="strength-control">
            <span class="strength-label">Strength</span>
            {#if items[index].strengthBatch}
              <input
                type="text"
                bind:value={items[index].strength}
                placeholder="e.g. -1, 0.5, 1..2:0.5"
                title="Sequence of strengths to sweep"
              />
            {:else}
              <input
                type="number"
                step={strengthStep}
                bind:value={items[index].strength}
                title="Negative values invert the effect, values above 1 exaggerate it"
              />
            {/if}
            <button
              type="button"
              class="batch-toggle"
              onclick={() => toggleStrengthBatch(items[index])}
              title="Toggle sequence mode"
            >
              {items[index].strengthBatch ? '−' : '+'}
            </button>
          </div>
          {#if items[index].strengthBatch}
            {@const sequenceError = numericSequenceError(String(items[index].strength ?? ''))}
            {#if sequenceError}
              <span class="strength-error">{sequenceError}</span>
            {/if}
          {/if}
        {/if}
        {#if itemExtra && items[index].node}
          {@render itemExtra(items[index], index)}
        {/if}
      </div>
    {/each}
  </div>
</div>

<style>
  .node-selector-list {
    margin-bottom: 1rem;
  }
  .header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }
  .header-row .list-title {
    margin: 0;
    font-weight: 500;
    color: var(--color-overlay-text);
  }
  .small-btn {
    min-width: 0;
    min-height: 0;
    padding: 0.25rem 0.75rem;
    font-size: 0.85rem;
  }
  .members-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .list-item-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--color-border-glass-1);
  }
  .list-item-container:last-child {
    border-bottom: none;
    padding-bottom: 0;
  }
  .member-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .member-selector {
    flex-grow: 1;
    min-width: 0;
  }
  .member-selector :global(.form-field) {
    margin-bottom: 0;
  }
  .strength-control {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-left: 0.25rem;
    padding-right: 2.25rem;
  }
  .strength-label {
    font-size: 0.85rem;
    color: var(--color-text-muted);
  }
  .strength-control input {
    flex-grow: 1;
    min-width: 0;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--color-overlay-border-primary, rgba(255, 255, 255, 0.1));
    color: var(--color-overlay-text);
    padding: 0.35rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
    box-sizing: border-box;
  }
  .strength-error {
    padding-left: 0.25rem;
    font-size: 0.75rem;
    color: var(--color-error);
  }
  .batch-toggle {
    flex-shrink: 0;
    background: none;
    border: 1px solid var(--color-overlay-border-primary, rgba(255, 255, 255, 0.2));
    color: var(--color-overlay-text);
    cursor: pointer;
    width: 20px;
    height: 20px;
    min-width: 20px;
    min-height: 20px;
    border-radius: 50%;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    transition: all 0.2s ease;
  }
  .batch-toggle:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: var(--color-overlay-text);
  }
  .remove-button {
    flex-shrink: 0;
    min-width: 0;
    min-height: 0;
    cursor: pointer;
    font-weight: bold;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    font-size: 12px;
    background-color: transparent;
    border: 1px solid var(--color-border-glass-1);
    color: var(--color-text-muted);
    transition: all 0.2s ease;
  }
  .remove-button:hover {
    background-color: var(--color-error);
    border-color: var(--color-error);
    color: white;
    transform: scale(1.05);
  }
  .remove-button:active {
    transform: scale(0.95);
  }
</style>
