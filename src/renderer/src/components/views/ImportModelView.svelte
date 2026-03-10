<!-- src/renderer/src/components/views/ImportModelView.svelte -->
<script lang="ts">
  import { onMount } from 'svelte'
  import type { FormConfig } from '../../utils/forms'
  import DynamicForm from '../DynamicForm.svelte'
  import { backendStatus } from '../../utils/stores'

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
  let formData: Record<string, unknown> = $state({})
  let inProgress = $state(false)
  let isFormValid = $state(false)
  let remoteModels = $state<string[]>([])
  let selectedRemoteModel = $state<string>('');
  let copyToStaging = $state(false);

  let runMode = $derived($backendStatus.run_mode)
  let currentEngineConfig = $derived(engines.find((e) => e.id === selectedEngine))

  $effect(async () => {
    if (runMode === 'unknown') {
      formFields = []
      return
    }

    inProgress = true
    try {
      if (runMode === 'remote') {
        remoteModels = await window.api.getModels()
        if (remoteModels.length > 0) {
          selectedRemoteModel = remoteModels[0]
        }
      } else if (runMode === 'local') {
        const config = await window.api.get_import_form_config(selectedEngine)
        if (config && Array.isArray(config)) {
          formFields = config
        } else {
          throw new Error('Invalid config format for import received from backend.')
        }
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
    if (runMode === 'local' && !isFormValid) return
    if (runMode === 'remote' && !selectedRemoteModel) return

    inProgress = true
    try {
      const payload =
        runMode === 'remote'
          ? {
              config_path: `models/${selectedRemoteModel}/model_config.json`,
              checkpoint_path: `models/${selectedRemoteModel}/model.ckpt`,
              // Also pass the name for display or other purposes
              name: selectedRemoteModel
            }
          : {
              ...formData,
              copy_to_staging: copyToStaging
            }

      await window.api.importModel({
        engine: selectedEngine,
        ...payload
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

  let isReadyToImport = $derived(
    runMode === 'remote' ? !!selectedRemoteModel : isFormValid
  );
</script>

<div class="view-container">
  <div class="view-content">
    <label>
      Engine:
      <select bind:value={selectedEngine} disabled={runMode === 'remote'}>
        {#each engines as engine (engine.id)}
          <option value={engine.id}>{engine.name}</option>
        {/each}
      </select>
    </label>

    {#if runMode === 'local'}
      {#if currentEngineConfig?.description}
        <p class="engine-description">{currentEngineConfig.description}</p>
      {/if}
      <DynamicForm config={formFields} bind:formData bind:isFormValid />

      <div class="inline-checkbox">
        <label for="copy-to-staging">Prepare for server export</label>
        <input type="checkbox" id="copy-to-staging" bind:checked={copyToStaging} />
        <p>
          If checked, this will copy the model files to a local `_exports` directory for manual
          upload to a remote server.
        </p>
      </div>
    {:else if runMode === 'remote'}
      <p class="engine-description">Select a pre-configured model from the backend.</p>
      {#if remoteModels.length > 0}
        <label>
          Available Models:
          <select bind:value={selectedRemoteModel}>
            {#each remoteModels as modelName (modelName)}
              <option value={modelName}>{modelName}</option>
            {/each}
          </select>
        </label>
      {:else}
        <div class="dropdown-empty">No remote models found on the backend.</div>
      {/if}
    {/if}
  </div>

  <div class="panel-actions">
    <button onclick={onclose}>Cancel</button>
    <button class="primary" onclick={importModel} disabled={!isReadyToImport || inProgress}>
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
  .dropdown-empty {
    padding: 1rem;
    text-align: center;
    color: var(--color-text-overlay-secondary);
    font-style: italic;
  }
  .inline-checkbox {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1rem;
  }
  .inline-checkbox label {
    margin-bottom: 0;
  }
  .inline-checkbox p {
    font-size: 0.8rem;
    color: var(--color-text-overlay-secondary);
  }
</style>
