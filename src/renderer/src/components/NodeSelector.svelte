<script lang="ts">
  import type { ModelData } from '../utils/forms'
  import { selectionStore } from '../utils/stores'

  type NodeData = ModelData | AudioData

  let { label, selectionType, node, id, onSelect } = $props<{
    label: string
    selectionType: 'model' | 'audio'
    node: NodeData | null
    id: string
    onSelect: (newNode: NodeData) => void
  }>()

  function selectNodeFromGraph(): void {
    selectionStore.startSelection(selectionType, node?.id ?? null, (selected) => {
      if (selected.type === selectionType) {
        onSelect(selected)
      }
    })
  }
</script>

<div class="form-field">
  <label for={id}>{label}</label>
  <div class="model-selector-wrapper">
    <input {id} type="text" readonly value={node ? node.alias || node.name : 'None selected'} />
    <button onclick={selectNodeFromGraph}> Select </button>
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
  }
  .model-selector-wrapper button {
    flex-shrink: 0;
  }
</style>
