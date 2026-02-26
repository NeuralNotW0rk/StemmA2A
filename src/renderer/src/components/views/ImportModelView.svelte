<!-- src/renderer/src/components/views/ImportModelView.svelte -->
<script lang="ts">
  import { onMount } from 'svelte'
  import type { FormConfig } from '../../utils/forms'
  import DynamicForm from '../DynamicForm.svelte'

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

  let { onclose, onrefresh, onError } = $props<{
    onclose: () => void
    onrefresh: () => void
    onError: (error: { title: string; message: string }) => void
  }>()

  let selectedEngine = $state(engines[0].id)
  let formFields = $state<FormConfig>([])
  let formData: Record<string, string> = $state({})
  let inProgress = $state(false)

  let currentEngineConfig = $derived(engines.find((e) => e.id === selectedEngine))
  let isFormValid = $derived(
    formFields.every(
      (f) => !f.validation?.required || (formData[f.name] && formData[f.name].trim() !== '')
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
      } catch (e) {
        console.error('Failed to load import form config:', e)
        const message = e instanceof Error ? e.message : String(e)
        onError({ title: 'Config Error', message })
      } finally {
        inProgress = false
      }
    }
  })

  async function importModel(): Promise<void> {
    if (!isFormValid) return

    inProgress = true
    try {
      await window.api.importModel({
        engine: selectedEngine,
        ...formData
      })
      formData = {}
      onclose()
      onrefresh()
    } catch (e) {
      console.error('Failed to import model:', e)
      const message = e instanceof Error ? e.message : String(e)
      onError({ title: 'Import Failed', message })
    } finally {
      inProgress = false
    }
  }
</script>

<div class="view-container">
  <div class="view-content">
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

    <DynamicForm config={formFields} bind:formData />
  </div>

  <div class="panel-actions">
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
  }
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
  select option {
    background: var(--color-background-medium);
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
