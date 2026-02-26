<script lang="ts">
  import { onMount, tick } from 'svelte'
  import type { FormConfig, ModelData, AudioData } from '../../utils/forms'
  import { selectionStore } from '../../utils/stores'
  import DynamicForm from '../DynamicForm.svelte'

  type NodeData = ModelData | AudioData

  let { node, onClose, onGenerate, onError } = $props<{
    node: NodeData
    onClose: () => void
    onGenerate: (data: unknown) => void
    onError: (error: { title: string; message: string }) => void
  }>()

  let formFields: FormConfig | null = $state(null)
  let formData: Record<string, any> = $state({})
  let isLoading = $state(true)
  let error: string | null = $state(null)
  let inProgress = $state(false)
  let selectedModel: ModelData | null = $state(null)
  let isVariation = $state(false)

  async function initialize() {
    try {
      isLoading = true
      error = null
      formFields = null
      formData = {}
      selectedModel = null
      isVariation = false

      await tick()

      let engine: string | null = null
      const title = node.type === 'model' ? 'Generate' : 'Variation'

      if (node.type === 'model') {
        selectedModel = node as ModelData
        engine = selectedModel.engine
        formData = { ...node }
      } else if (node.type === 'audio') {
        isVariation = true
        formData = { init_audio: node.name }
      }

      if (engine) {
        const config = await window.api.getEngineConfig(engine)
        if (config && config.generate && Array.isArray(config.generate)) {
          formFields = config.generate
          const newFormData = { ...formData }
          for (const field of formFields) {
            if (newFormData[field.name] === undefined || newFormData[field.name] === null) {
              newFormData[field.name] = field.defaultValue as string | number | boolean | null
            }
          }
          formData = newFormData
        } else {
          throw new Error('Invalid config format received from backend.')
        }
      }
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : 'Failed to load configuration.'
      console.error('Failed to fetch form config:', e)
      error = message
      onError({ title: 'Configuration Error', message })
    } finally {
      isLoading = false
    }
  }

  $effect(() => {
    if (node) {
      initialize()
    }
  })

  function selectModelFromGraph() {
    selectionStore.startSelection('model', (selected) => {
      if (selected.type === 'model') {
        selectedModel = selected as ModelData
        initialize()
      }
    })
  }

  async function generate(): Promise<void> {
    const title = isVariation ? 'Variation' : 'Generation'
    if (!selectedModel) {
      onError({ title: title + ' Failed', message: 'A model must be selected for generation.' })
      return
    }

    const payload = {
      ...formData,
      model_name: selectedModel.name
    }
    inProgress = true
    try {
      const result = await window.api.generate(payload)
      onGenerate(result)
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e)
      onError({ title: title + ' Failed', message })
    } finally {
      inProgress = false
    }
  }
</script>

<div class="view-container">
  <div class="view-content">
    {#if isVariation && !selectedModel}
      <div class="model-selection">
        <p>Select a model to use for the variation.</p>
        <button onclick={selectModelFromGraph}>Select from Graph</button>
      </div>
    {/if}
    {#if isLoading}
      <p>Loading configuration...</p>
    {:else if error}
      <div class="error-message">
        <p>Error: {error}</p>
      </div>
    {:else if formFields}
      <DynamicForm config={formFields} bind:formData />
    {:else if !isVariation}
      <p>Could not load generation form.</p>
    {/if}
  </div>

  <div class="panel-actions">
    <button onclick={onClose}>Cancel</button>
    <button class="primary" onclick={generate} disabled={isLoading || !!error || !selectedModel}>
      {#if inProgress}
        <div class="spinner"></div>
      {:else}
        Generate
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
  .model-selection {
    padding: 1rem;
    text-align: center;
  }
  .model-selection button {
    margin-top: 1rem;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    background: var(--color-primary);
    border: 1px solid var(--color-primary);
    color: var(--color-overlay-text);
  }

  .error-message {
    padding: 1rem;
    color: var(--color-error);
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
