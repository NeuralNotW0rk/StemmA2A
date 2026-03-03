<script lang="ts">
  import { tick } from 'svelte'
  import type { FormConfig, ModelData, AudioData } from '../../utils/forms'
  import { selectionStore, activeNodeStore } from '../../utils/stores'
  import DynamicForm from '../DynamicForm.svelte'
  import NodeSelector from '../NodeSelector.svelte'

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
  let lastNodeId: string | null = $state(null)

  async function loadEngineConfig(engine: string) {
    try {
      isLoading = true
      error = null
      formFields = null
      await tick()

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
    if (node && node.id !== lastNodeId) {
      lastNodeId = node.id
      // Reset component state when the input node changes
      error = null
      inProgress = false

      if (node.type === 'model') {
        isVariation = false
        selectedModel = node as ModelData
        // Base form data on the node, existing data is merged in loadEngineConfig
        formData = { ...node }
        if (selectedModel.engine) {
          loadEngineConfig(selectedModel.engine)
        } else {
          isLoading = false
          formFields = null
        }
      } else if (node.type === 'audio') {
        isVariation = true
        selectedModel = null
        formFields = null
        formData = { init_audio: node.name }
        isLoading = false // Not loading engine config until a model is selected
      }
    }
  })

  $effect(() => {
    if (selectedModel) {
      console.log('GenerationView: Setting active node store to:', selectedModel.id)
      activeNodeStore.set(selectedModel)
    } else {
      console.log('GenerationView: Setting active node store to (from node prop):', node?.id)
      activeNodeStore.set(node)
    }
  })

  function handleModelSelect(newNode: ModelData) {
    const newModel = newNode as ModelData
    const oldEngine = selectedModel?.engine
    selectedModel = newModel

    if (newModel.engine !== oldEngine) {
      // If the engine is different, we need to load new config.
      // Keep existing form data where field names overlap.
      loadEngineConfig(newModel.engine)
    }
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
    <NodeSelector
      label="Model"
      selectionType="model"
      node={selectedModel}
      id="model-selector"
      onSelect={handleModelSelect}
    />

    {#if isLoading}
      <p>Loading configuration...</p>
    {:else if error}
      <div class="error-message">
        <p>Error: {error}</p>
      </div>
    {:else if formFields}
      <DynamicForm config={formFields} bind:formData />
    {:else if isVariation}
      <p class="centered-text">Select a model to see its generation options.</p>
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
  .centered-text {
    text-align: center;
    padding: 2rem 1rem;
    color: var(--color-text-muted);
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
