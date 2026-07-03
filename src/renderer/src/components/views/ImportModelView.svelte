<!-- src/renderer/src/components/views/ImportModelView.svelte -->
<script lang="ts">
  import type { FormConfig } from '../../utils/forms'
  import type { ErrorInfo } from '../../utils/types'
  import DynamicForm from '../DynamicForm.svelte'

  interface AdapterConfig {
    id: string
    name: string
    description: string
  }

  export const adapters: Omit<AdapterConfig, 'fields'>[] = [
    {
      id: 'stable_audio_tools',
      name: 'Stable Audio Tools',
      description: 'Manually choose any Stable Audio-based model checkpoint and config file'
    }
  ]

  let { onclose, onrefresh, onError } = $props<{
    onclose: () => void
    onrefresh: () => void
    onError: (error: ErrorInfo) => void
  }>()

  let selectedAdapter = $state(adapters[0].id)
  let formFields = $state<FormConfig>([])
  let formData: Record<string, unknown> = $state({})
  let inProgress = $state(false)
  let isFormValid = $state(false)

  // Shared Models state
  let importMode = $state<'shared' | 'local'>('shared')
  let sharedModels = $state<any[]>([])
  let selectedSharedModelId = $state('')
  let customModelName = $state('')

  let currentAdapterConfig = $derived(adapters.find((e) => e.id === selectedAdapter))
  let isImportValid = $derived(
    importMode === 'shared'
      ? selectedSharedModelId !== '' && customModelName.trim() !== ''
      : isFormValid
  )

  $effect(() => {
    const loadSharedModels = async (): Promise<void> => {
      try {
        const models = await window.api.getSharedModels()
        sharedModels = models
        if (models && models.length > 0) {
          selectedSharedModelId = models[0].id
          importMode = 'shared'
        } else {
          importMode = 'local'
        }
      } catch (e) {
        console.error('Failed to load shared models:', e)
        importMode = 'local'
      }
    }
    loadSharedModels()
  })

  $effect(() => {
    const model = sharedModels.find((m) => m.id === selectedSharedModelId)
    if (model) {
      customModelName = model.name
    }
  })

  $effect(() => {
    const loadConfig = async (): Promise<void> => {
      inProgress = true
      try {
        const config = await window.api.get_import_form_config(selectedAdapter)
        if (config && Array.isArray(config)) {
          formFields = config
        } else {
          throw new Error('Invalid config format for import received from backend.')
        }
      } catch (e: unknown) {
        console.error('Failed to load import config:', e)
        const message = e instanceof Error ? e.message : String(e)
        onError({ title: 'Config Error', message })
      } finally {
        inProgress = false
      }
    }
    loadConfig()
  })

  async function importModel(): Promise<void> {
    if (!isImportValid) return

    inProgress = true
    try {
      if (importMode === 'shared') {
        const selectedModel = sharedModels.find((m) => m.id === selectedSharedModelId)
        if (!selectedModel) throw new Error('Selected shared model not found')

        await window.api.importModel({
          model_element: {
            ...selectedModel,
            name: customModelName
          }
        })
      } else {
        await window.api.importModel({
          adapter: selectedAdapter,
          ...formData
        })
      }
      formData = {}
      onclose()
      onrefresh()
    } catch (e: unknown) {
      console.error('Failed to import model:', e)
      const message = e instanceof Error ? e.message : String(e)
      onError({ title: 'Import Failed', message })
    } finally {
      inProgress = false
    }
  }
</script>

<div class="view-container">
  {#if sharedModels.length > 0}
    <div class="tabs-container">
      <button
        class="tab-btn {importMode === 'shared' ? 'active' : ''}"
        onclick={() => (importMode = 'shared')}
        type="button"
      >
        Shared Models
      </button>
      <button
        class="tab-btn {importMode === 'local' ? 'active' : ''}"
        onclick={() => (importMode = 'local')}
        type="button"
      >
        Local Files
      </button>
    </div>
  {/if}

  <div class="view-content">
    {#if importMode === 'shared'}
      <label>
        Shared Model:
        <select bind:value={selectedSharedModelId}>
          {#each sharedModels as model (model.id)}
            <option value={model.id}>{model.name} ({model.model_type})</option>
          {/each}
        </select>
      </label>

      <label style="margin-top: 1.25rem; display: block;">
        Model Instance Name:
        <input type="text" bind:value={customModelName} placeholder="Custom Model Name" />
      </label>

      {#if selectedSharedModelId}
        {@const selectedModel = sharedModels.find((m) => m.id === selectedSharedModelId)}
        {#if selectedModel}
          <div class="model-details">
            <h4>Model Details</h4>
            <div class="detail-row">
              <span class="detail-label">ID:</span>
              <span class="detail-value">{selectedModel.id}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Adapter:</span>
              <span class="detail-value">{selectedModel.adapter}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Type:</span>
              <span class="detail-value">{selectedModel.model_type}</span>
            </div>
            {#if selectedModel.checkpoint?.path}
              <div class="detail-row">
                <span class="detail-label">Checkpoint:</span>
                <span class="detail-value path-value" title={selectedModel.checkpoint.path}>
                  {selectedModel.checkpoint.path}
                </span>
              </div>
            {/if}
          </div>
        {/if}
      {/if}
    {:else}
      <label>
        Adapter:
        <select bind:value={selectedAdapter}>
          {#each adapters as adapter (adapter.id)}
            <option value={adapter.id}>{adapter.name}</option>
          {/each}
        </select>
      </label>

      {#if currentAdapterConfig?.description}
        <p class="adapter-description">{currentAdapterConfig.description}</p>
      {/if}
      <DynamicForm config={formFields} bind:formData bind:isFormValid />
    {/if}
  </div>

  <div class="panel-actions">
    <button onclick={onclose}>Cancel</button>
    <button class="primary" onclick={importModel} disabled={!isImportValid || inProgress}>
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
  .tabs-container {
    display: flex;
    gap: 0.5rem;
    padding: 0.75rem 1rem 0 1rem;
    border-bottom: 1px solid var(--color-border-glass-1);
    background-color: var(--color-background-glass-2);
  }
  .tab-btn {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--color-text-overlay-secondary);
    padding: 0.5rem 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .tab-btn:hover {
    color: var(--color-text-overlay-primary);
  }
  .tab-btn.active {
    color: var(--color-overlay-text);
    border-bottom-color: var(--color-primary);
  }
  .view-content {
    flex-grow: 1;
    overflow-y: auto;
    padding: 1rem;
  }
  .adapter-description {
    font-size: 0.875rem;
    color: var(--color-text-overlay-secondary);
    margin-top: 0.75rem;
    margin-bottom: 0.5rem;
    padding: 0.75rem;
    background-color: var(--color-background-glass-1);
    border-radius: 0.375rem;
    border: 1px solid var(--color-border-glass-1);
  }
  label {
    display: block;
    color: var(--color-overlay-text);
    font-weight: 500;
    margin-bottom: 0.5rem;
  }
  select {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
  }
  select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  select option {
    background: var(--color-background-medium);
  }
  input[type='text'] {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
    margin-top: 0.25rem;
    font-family: inherit;
    font-size: 0.875rem;
  }
  input[type='text']:focus {
    outline: none;
    border-color: var(--color-primary);
    background: var(--color-background-glass-1);
  }
  .model-details {
    margin-top: 1.5rem;
    padding: 1rem;
    background-color: var(--color-background-glass-1);
    border: 1px solid var(--color-border-glass-1);
    border-radius: 0.5rem;
  }
  .model-details h4 {
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--color-text-overlay-primary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .detail-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.825rem;
    padding: 0.375rem 0;
    border-bottom: 1px dashed var(--color-border-glass-1);
  }
  .detail-row:last-child {
    border-bottom: none;
  }
  .detail-label {
    color: var(--color-text-overlay-secondary);
  }
  .detail-value {
    color: var(--color-overlay-text);
    font-weight: 500;
  }
  .path-value {
    max-width: 70%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
