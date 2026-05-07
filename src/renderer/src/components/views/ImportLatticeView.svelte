<!-- src/renderer/src/components/views/ImportLatticeView.svelte -->
<script lang="ts">
  import { initiatorNodeStore } from '../../utils/stores'
  import NodeSelector from '../NodeSelector.svelte'
  import type { NodeData } from '../../utils/forms'

  let { onclose, onrefresh, onError } = $props<{
    onclose: () => void
    onrefresh: () => void
    onError: (error: { title: string; message: string }) => void
  }>()

  let name = $state('')
  let checkpointPath = $state('')
  let inProgress = $state(false)

  let baseModel = $state<NodeData | null>($initiatorNodeStore as NodeData | null)

  let isFormValid = $derived(name.trim() !== '' && checkpointPath.trim() !== '' && !!baseModel)

  async function selectFile(): Promise<void> {
    const path = await window.api.openFile({
      title: 'Select Lattice File',
      filters: [{ name: 'SafeTensors', extensions: ['safetensors'] }]
    })
    if (path) {
      checkpointPath = path
      if (!name) {
        // Auto-fill name from filename, removing extension
        name =
          path
            .split(/[/\\]/)
            .pop()
            ?.replace(/\.[^/.]+$/, '') || ''
      }
    }
  }

  async function importLattice(): Promise<void> {
    if (!isFormValid || !baseModel) return

    inProgress = true
    try {
      await window.api.registerLattice({
        name: name,
        checkpoint_path: checkpointPath,
        base_model_id: baseModel.id
      })
      onclose()
      onrefresh()
    } catch (e) {
      console.error('Failed to import lattice:', e)
      const message = e instanceof Error ? e.message : String(e)
      onError({ title: 'Import Failed', message })
    } finally {
      inProgress = false
    }
  }
</script>

<div class="view-container">
  <div class="view-content">
    <NodeSelector
      label="Base Model"
      filter={{ type: 'model' }}
      bind:node={baseModel}
      id="base-model-selector"
    />
    <label>
      Lattice Name
      <input type="text" bind:value={name} placeholder="e.g., My LoRA" />
    </label>
    <label>
      Checkpoint File
      <div class="file-input-container">
        <input
          type="text"
          bind:value={checkpointPath}
          placeholder="Select a .safetensors file..."
          readonly
        />
        <button onclick={selectFile}>Browse</button>
      </div>
    </label>
  </div>

  <div class="panel-actions">
    <button onclick={onclose}>Cancel</button>
    <button class="primary" onclick={importLattice} disabled={!isFormValid || inProgress}>
      {#if inProgress}
        <div class="spinner"></div>
      {:else}
        Import
      {/if}
    </button>
  </div>
</div>

<style>
  .view-container {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  .view-content {
    flex-grow: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  label {
    display: block;
    color: var(--color-overlay-text);
    font-weight: 500;
  }
  input {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
    margin-top: 0.25rem;
  }
  .file-input-container {
    display: flex;
    gap: 0.5rem;
  }
  .file-input-container input {
    flex-grow: 1;
  }
  .panel-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    padding: 1rem;
    border-top: 1px solid var(--color-border-glass-1);
    background-color: var(--color-background-glass-2);
    flex-shrink: 0;
  }
</style>
