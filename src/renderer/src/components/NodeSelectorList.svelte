<!-- src/renderer/src/components/NodeSelectorList.svelte -->
<script lang="ts">
  import { onDestroy, type Snippet } from 'svelte'
  import NodeSelector from './NodeSelector.svelte'
  import { selectionStore } from '../utils/stores'
  import type { NodeData } from '../utils/forms'
  import type { NodeFilter } from '../utils/types'

  export interface NodeListItem {
    id: number | string
    node: NodeData | string | null
    strength?: number
    [key: string]: any
  }

  let {
    title = 'Items',
    addButtonText = 'Add',
    filter = {},
    items = $bindable([]),
    idPrefix = 'list-item',
    minItems = 0,
    showStrengths = false,
    strengthMin = 0.0,
    strengthMax = 1.0,
    strengthStep = 0.01,
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
    strengthMin?: number
    strengthMax?: number
    strengthStep?: number
    defaultStrength?: number
    onAdd?: () => void
    itemExtra?: Snippet<[any, number]>
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
        const item = items.find((i) => i.id === newItem.id)
        if (item) {
          item.node = selected as NodeData
        }
      })
    }
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
            <NodeSelector {filter} bind:node={items[index].node} id={`${idPrefix}-${item.id}`} />
          </div>
          {#if index >= minItems}
            <button
              type="button"
              class="remove-button"
              onclick={() => handleRemove(item.id)}
              title="Remove item"
              aria-label="Remove item"
            >
              ✕
            </button>
          {/if}
        </div>
        {#if showStrengths && items[index].node}
          <div class="strength-control">
            <span class="strength-label">Strength</span>
            <input
              type="range"
              min={strengthMin}
              max={strengthMax}
              step={strengthStep}
              bind:value={items[index].strength}
              title={`Strength: ${items[index].strength}`}
            />
            <span class="strength-value">{Number(items[index].strength).toFixed(2)}</span>
          </div>
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
  .strength-control input[type="range"] {
    flex-grow: 1;
    accent-color: var(--color-primary);
  }
  .strength-value {
    font-size: 0.85rem;
    min-width: 2.5rem;
    text-align: right;
    color: var(--color-text-muted);
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
