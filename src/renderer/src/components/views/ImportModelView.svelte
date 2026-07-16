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

  interface SharedModel {
    id: string
    name: string
    adapter: string
    model_type: string
    checkpoint?: {
      path?: string
    }
  }

  export const adapters: AdapterConfig[] = [
    {
      id: 'stable_audio_tools',
      name: 'Stable Audio Tools',
      description: 'Manually choose any Stable Audio-based model checkpoint and config file'
    },
    {
      id: 'stylegan2',
      name: 'StyleGAN2',
      description: 'Choose a StyleGAN2 model checkpoint (.pkl or .pt)'
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
  let importMode = $state<'shared' | 'local'>('local')
  let sharedModels = $state<SharedModel[]>([])
  let selectedSharedModelId = $state('')
  let customModelName = $state('')
  let isLoadingShared = $state(true)
  let loadError = $state<string | null>(null)

  let currentAdapterConfig = $derived(
    adapters.find((e: AdapterConfig): boolean => e.id === selectedAdapter)
  )
  let isImportValid = $derived(
    importMode === 'shared'
      ? selectedSharedModelId !== '' && customModelName.trim() !== ''
      : isFormValid
  )

  $effect((): void => {
    const loadSharedModels = async (): Promise<void> => {
      isLoadingShared = true
      loadError = null
      try {
        const models = (await window.api.getSharedModels()) as SharedModel[]
        sharedModels = models
        if (models && models.length > 0) {
          selectedSharedModelId = models[0].id
        }
      } catch (e: unknown) {
        console.error('Failed to load shared models:', e)
        loadError = e instanceof Error ? e.message : String(e)
      } finally {
        isLoadingShared = false
      }
    }
    loadSharedModels()
  })

  $effect((): void => {
    const model = sharedModels.find((m: SharedModel): boolean => m.id === selectedSharedModelId)
    if (model) {
      customModelName = model.name
    }
  })

  $effect((): void => {
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
        const selectedModel = sharedModels.find((m: SharedModel): boolean => m.id === selectedSharedModelId)
        if (!selectedModel) throw new Error('Selected shared model not found')

        await window.api.importModel({
          model_element: {
            ...$state.snapshot(selectedModel),
            name: customModelName
          }
        })
      } else {
        await window.api.importModel({
          adapter: selectedAdapter,
          ...$state.snapshot(formData)
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
  <div class="tabs-container">
    <button
      class="tab-btn {importMode === 'local' ? 'active' : ''}"
      onclick={(): void => {
        importMode = 'local'
      }}
      type="button"
    >
      Local Files
    </button>
    <button
      class="tab-btn {importMode === 'shared' ? 'active' : ''}"
      onclick={(): void => {
        importMode = 'shared'
      }}
      type="button"
    >
      Shared Models
    </button>
  </div>

  <div class="view-content">
    {#if importMode === 'shared'}
      {#if isLoadingShared}
        <div class="loading-container">
          <div class="spinner"></div>
          <span>Fetching shared models from remote server...</span>
        </div>
      {:else if loadError}
        <div class="error-container">
          <div class="error-title">Failed to Fetch Shared Models</div>
          <div class="error-message">{loadError}</div>
          <div class="error-action">
            Verify the connection to your remote engine (configured via <code>ENGINE_URL</code> in your <code>.env</code> file).
          </div>
        </div>
      {:else if sharedModels.length === 0}
        <div class="no-models-container">
          <div class="no-models-icon">☁️</div>
          <div class="no-models-title">No Shared Models Available</div>
          <div class="no-models-desc">
            The remote server did not return any shared models.
          </div>
          <div class="instructions-card">
            <h5>How to share models:</h5>
            <ol>
              <li>Create or edit <code>shared_models.json</code> in the backend directory of your remote engine.</li>
              <li>Add entries mapping model configurations (checkpoint and config files).</li>
              <li>Ensure the files actually exist on the remote host's filesystem.</li>
            </ol>
          </div>
          <button
            class="switch-btn"
            onclick={(): void => {
              importMode = 'local'
            }}
            type="button"
          >
            Use Local Files Instead
          </button>
        </div>
      {:else}
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
          {@const selectedModel = sharedModels.find((m: SharedModel): boolean => m.id === selectedSharedModelId)}
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
      <DynamicForm config={formFields} bind:formData bind:isFormValid allowSequences={false} />
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

  .loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
    gap: 1rem;
    color: var(--color-text-overlay-secondary);
  }
  .spinner {
    width: 2rem;
    height: 2rem;
    border: 3px solid var(--color-border-glass-1);
    border-top-color: var(--color-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-container,
  .no-models-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 2rem 1.5rem;
    background-color: var(--color-background-glass-1);
    border: 1px solid var(--color-border-glass-1);
    border-radius: 0.75rem;
    margin-top: 1rem;
  }

  .error-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #ef4444;
    margin-bottom: 0.5rem;
  }

  .error-message {
    font-size: 0.875rem;
    color: var(--color-text-overlay-secondary);
    background-color: rgba(239, 68, 68, 0.1);
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    border: 1px solid rgba(239, 68, 68, 0.2);
    font-family: monospace;
    word-break: break-all;
    margin-bottom: 1rem;
    width: 100%;
  }

  .error-action {
    font-size: 0.825rem;
    color: var(--color-text-overlay-secondary);
  }

  .no-models-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
  }

  .no-models-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--color-overlay-text);
    margin-bottom: 0.5rem;
  }

  .no-models-desc {
    font-size: 0.875rem;
    color: var(--color-text-overlay-secondary);
    margin-bottom: 1.5rem;
    max-width: 24rem;
  }

  .instructions-card {
    text-align: left;
    width: 100%;
    background-color: var(--color-background-glass-2);
    border: 1px solid var(--color-border-glass-1);
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1.5rem;
  }

  .instructions-card h5 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--color-text-overlay-primary);
  }

  .instructions-card ol {
    margin: 0;
    padding-left: 1.25rem;
    font-size: 0.825rem;
    color: var(--color-text-overlay-secondary);
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .instructions-card code {
    background-color: rgba(255, 255, 255, 0.08);
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    font-family: monospace;
    font-size: 0.8rem;
  }

  .switch-btn {
    background-color: var(--color-primary);
    color: #ffffff;
    border: none;
    padding: 0.5rem 1.25rem;
    border-radius: 0.375rem;
    font-weight: 500;
    font-size: 0.875rem;
    cursor: pointer;
    transition: background-color 0.2s ease;
  }

  .switch-btn:hover {
    background-color: var(--color-primary-hover);
  }
</style>
