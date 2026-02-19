<!-- src/renderer/src/components/ImportModelDialog.svelte -->
<script lang="ts">
  import ErrorDialog from './ErrorDialog.svelte'

  export interface EngineField {
    key: string
    label: string
    type: 'file' | 'text'
    filters?: { name: string; extensions: string[] }[]
    placeholder?: string
    required?: boolean
  }

  export interface EngineConfig {
    id: string
    name: string
    description: string
    fields: EngineField[]
  }

  export const engines: EngineConfig[] = [
    {
      id: 'stable_audio_tools',
      name: 'Stable Audio Tools',
      description: 'Manually choose any Stable Audio-based model checkpoint and config file',
      fields: [
        {
          key: 'name',
          label: 'Name',
          type: 'text',
          required: true
        },
        {
          key: 'checkpoint_path',
          label: 'Checkpoint Path',
          type: 'file',
          filters: [
            { name: 'Model Files', extensions: ['ckpt', 'safetensors', 'pt', 'pth', 'bin'] },
            { name: 'All Files', extensions: ['*'] }
          ],
          placeholder: '/path/to/model.ckpt',
          required: true
        },
        {
          key: 'config_path',
          label: 'Config Path (model_config.json)',
          type: 'file',
          filters: [{ name: 'JSON Files', extensions: ['json'] }],
          placeholder: '/path/to/model_config.json',
          required: true
        }
      ]
    }
  ]

  interface Props {
    show: boolean
    onclose: () => void
    onrefresh: () => void
  }

  let { show, onclose, onrefresh }: Props = $props()

  let selectedEngine = $state(engines[0].id)
  let engineFields: Record<string, string> = $state({})
  let showErrorDialog = $state(false)
  let errorMessage = $state('')
  let inProgress = $state(false)

  let currentEngineConfig = $derived(engines.find((e) => e.id === selectedEngine))
  let isFormValid = $derived(
    currentEngineConfig
      ? currentEngineConfig.fields.every(
          (f) => !f.required || (engineFields[f.key] && engineFields[f.key].trim() !== '')
        )
      : false
  )

  function closeDialog(): void {
    if (inProgress) return
    onclose()
  }

  async function importModel(): Promise<void> {
    if (currentEngineConfig) {
      for (const field of currentEngineConfig.fields) {
        if (field.required && !engineFields[field.key]?.trim()) return
      }
    }

    inProgress = true
    try {
      await window.api.importModel({
        engine: selectedEngine,
        ...engineFields
      })
      engineFields = {}
      onclose()
      onrefresh()
    } catch (error) {
      console.error('Failed to import model:', error)
      if (error instanceof Error) {
        errorMessage = error.message
      } else {
        errorMessage = String(error)
      }
      showErrorDialog = true
    } finally {
      inProgress = false
    }
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      closeDialog()
    }
  }

  async function selectFieldFile(
    key: string,
    filters?: { name: string; extensions: string[] }[]
  ): Promise<void> {
    const path = await window.api.openFile({ title: 'Select File', filters })
    if (path) {
      engineFields[key] = path
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if show}
  <div
    class="dialog-overlay"
    role="button"
    tabindex="0"
    aria-label="Close dialog"
    onclick={(e) => {
      if (e.target === e.currentTarget) closeDialog()
    }}
    onkeydown={(e) => {
      if (e.key === 'Enter') closeDialog()
    }}
  >
    <div
      class="dialog"
      role="dialog"
      aria-modal="true"
      aria-labelledby="dialog-title"
      tabindex="-1"
    >
      <div class="dialog-header">
        <h3 id="dialog-title">Import Model</h3>
        <button onclick={onclose}>×</button>
      </div>

      <div class="dialog-content">
        <label>
          Engine:
          <select bind:value={selectedEngine}>
            {#each engines as engine (engine.id)}
              <option value={engine.id}>{engine.name}</option>
            {/each}
          </select>
        </label>

        {#if currentEngineConfig?.description}
          <p class="engine-description">{currentEngineConfig.description}</p>
        {/if}

        {#each engines.find((e) => e.id === selectedEngine)?.fields || [] as field (field.key)}
          <label style="margin-top: 1rem;">
            {field.label}:
            {#if field.type === 'file'}
              <div class="path-input">
                <input
                  type="text"
                  bind:value={engineFields[field.key]}
                  placeholder={field.placeholder}
                />
                <button onclick={() => selectFieldFile(field.key, field.filters)}>Browse</button>
              </div>
            {:else}
              <input
                type="text"
                bind:value={engineFields[field.key]}
                placeholder={field.placeholder}
              />
            {/if}
          </label>
        {/each}
      </div>

      <div class="dialog-actions">
        <button onclick={onclose}>Cancel</button>
        <button class="primary" onclick={importModel} disabled={!isFormValid || inProgress}>
          {#if inProgress}
            <div class="spinner"></div>
          {:else}
            Import
          {/if}
        </button>
      </div>
    </div>
  </div>
{/if}

<ErrorDialog
  show={showErrorDialog}
  title="Import Failed"
  message={errorMessage}
  onclose={() => (showErrorDialog = false)}
/>

<style>
  .engine-description {
    font-size: 0.875rem;
    color: var(--color-text-overlay-secondary);
    margin-top: 0.75rem;
    margin-bottom: 0.5rem;
    padding: 0.75rem;
    background-color: var(--color-background-glass-1);
    border-radius: 0.375rem;
    border: 1px solid var(--color-border-glass-1);
  }

  .dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--color-overlay-background-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    backdrop-filter: blur(4px);
  }

  .dialog {
    background: var(--color-background-medium);
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.5rem;
    min-width: 400px;
    max-width: 500px;
  }

  .dialog-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--color-border-glass-1);
  }

  .dialog-header h3 {
    margin: 0;
    color: var(--color-overlay-text);
  }

  .dialog-header button {
    background: none;
    border: none;
    color: var(--color-overlay-text);
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .dialog-content {
    padding: 1.5rem;
  }

  .dialog-content label {
    display: block;
    color: var(--color-overlay-text);
    font-weight: 500;
    margin-bottom: 0.5rem;
  }

  .dialog-content select {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
  }

  .dialog-content select option {
    background: var(--color-background-medium);
  }

  .dialog-content input[type='text'] {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
    box-sizing: border-box;
  }

  .path-input {
    display: flex;
    gap: 0.5rem;
  }

  .path-input input {
    flex: 1;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
  }

  .path-input button {
    background: var(--color-primary-t-30);
    border: 1px solid var(--color-primary-t-50);
    color: var(--color-overlay-text);
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
  }

  .dialog-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--color-border-glass-1);
  }

  .dialog-actions button {
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
    min-width: 80px;
    min-height: 38px;
  }

  .dialog-actions button:not(.primary) {
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
  }

  .dialog-actions button.primary {
    background: var(--color-primary);
    border: 1px solid var(--color-primary);
    color: var(--color-overlay-text);
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .dialog-actions button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinner {
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-left-color: #fff;
    border-radius: 50%;
    width: 18px;
    height: 18px;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
