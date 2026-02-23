<script lang="ts">
  import { onMount } from 'svelte'
  import type { FormConfig } from '../../utils/forms'
  import DynamicForm from '../DynamicForm.svelte'
  import ErrorDialog from './ErrorDialog.svelte'

  interface EngineConfig {
    id: string
    name: string
    description: string
  }

  export const engines: Omit<EngineConfig, 'fields'>[] = [
    {
      id: 'stable_audio_tools',
      name: 'Stable Audio Tools',
      description: 'Manually choose any Stable Audio-based model checkpoint and config file'
    }
  ]

  interface Props {
    onclose: () => void
    onrefresh: () => void
  }

  let { onclose, onrefresh }: Props = $props()

  let selectedEngine = $state(engines[0].id)
  let formFields = $state<FormConfig>([])
  let engineFields: Record<string, string> = $state({})
  let showErrorDialog = $state(false)
  let errorMessage = $state('')
  let inProgress = $state(false)

  let currentEngineConfig = $derived(engines.find((e) => e.id === selectedEngine))
  let isFormValid = $derived(
    formFields.every(
      (f) => !f.validation?.required || (engineFields[f.name] && engineFields[f.name].trim() !== '')
    )
  )

  onMount(async () => {
    if (selectedEngine) {
      try {
        inProgress = true
        const config = await window.api.getEngineConfig(selectedEngine)
        if (config && config.import && Array.isArray(config.import)) {
          formFields = config.import
        } else {
          throw new Error('Invalid config format for import received from backend.')
        }
      } catch (error) {
        console.error('Failed to load import form config:', error)
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
  })

  function closeDialog(): void {
    if (inProgress) return
    onclose()
  }

  async function importModel(): Promise<void> {
    if (!isFormValid) return

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
</script>

<svelte:window onkeydown={handleKeydown} />

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
  <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="dialog-title" tabindex="-1">
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

      <DynamicForm config={formFields} bind:formData={engineFields} />
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
