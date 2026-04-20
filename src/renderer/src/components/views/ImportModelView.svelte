<!-- src/renderer/src/components/views/ImportModelView.svelte -->
<script lang="ts">
  import type { FormConfig } from '../../utils/forms'
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
    onError: (error: { title: string; message: string }) => void
  }>()

  let selectedAdapter = $state(adapters[0].id)
  let formFields = $state<FormConfig>([])
  let formData: Record<string, unknown> = $state({})
  let inProgress = $state(false)
  let isFormValid = $state(false)
  
  let currentAdapterConfig = $derived(adapters.find((e) => e.id === selectedAdapter))

  $effect(async () => {
    inProgress = true
    try {
      const config = await window.api.get_import_form_config(selectedAdapter)
      if (config && Array.isArray(config)) {
        formFields = config
      } else {
        throw new Error('Invalid config format for import received from backend.')
      }
    } catch (e) {
      console.error('Failed to load import config:', e)
      const message = e instanceof Error ? e.message : String(e)
      onError({ title: 'Config Error', message })
    } finally {
      inProgress = false
    }
  })

  async function importModel(): Promise<void> {
    if (!isFormValid) return

    inProgress = true
    try {
      await window.api.importModel({
        adapter: selectedAdapter,
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
