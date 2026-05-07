<!-- src/renderer/src/components/NodeSelectorList.svelte -->
<script lang="ts">
  import { onDestroy } from 'svelte'
  import NodeSelector from './NodeSelector.svelte'
  import { selectionStore } from '../utils/stores'

  export interface NodeListItem {
    id: number | string
    node: any
  }

  let {
    title = 'Items',
    addButtonText = 'Add',
    filter = {},
    items = $bindable([]),
    idPrefix = 'list-item',
    minItems = 0,
    onAdd
  } = $props<{
    title?: string
    addButtonText?: string
    filter?: Record<string, string | number | boolean>
    items: NodeListItem[]
    idPrefix?: string
    minItems?: number
    onAdd?: () => void
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
      const newItem = { id: nextId++, node: null }
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
          item.node = selected
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
      <label>{title}</label>
    {:else}
      <div></div>
    {/if}
    <button type="button" onclick={handleAdd} class="small-btn">{addButtonText}</button>
  </div>
  <div class="members-list">
    {#each items as item, index (item.id)}
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
    {/each}
  </div>
</div>

<style>
  .node-selector-list { margin-bottom: 1rem; }
  .header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
  .header-row label { margin: 0; font-weight: 500; color: var(--color-overlay-text); }
  .small-btn { min-width: 0; min-height: 0; padding: 0.25rem 0.75rem; font-size: 0.85rem; }
  .members-list { display: flex; flex-direction: column; gap: 0.5rem; }
  .member-row { display: flex; align-items: center; gap: 0.5rem; }
  .member-selector { flex-grow: 1; min-width: 0; }
  .member-selector :global(.form-field) { margin-bottom: 0; }
  .remove-button {
    flex-shrink: 0; min-width: 0; min-height: 0; cursor: pointer;
    font-weight: bold; width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    padding: 0; font-size: 12px; background-color: transparent;
    border: 1px solid var(--color-border-glass-1); color: var(--color-text-muted);
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