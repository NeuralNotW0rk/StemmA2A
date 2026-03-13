<script lang="ts">
  import { tick } from 'svelte'
  import type { FormConfig, ModelData } from '../../utils/forms'
  import { activeNodeStore } from '../../utils/stores'
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
  let formData: Record<string, unknown> = $state({})
  let isLoading = $state(true)
  let error: string | null = $state(null)
  let inProgress = $state(false)
  let selectedModel: ModelData | null = $state(null)
  let lastNodeId: string | null = $state(null)
  let isFormValid = $state(false)

  async function loadAdapterConfig(adapter: string): Promise<void> {
    try {
      isLoading = true
      error = null
      formFields = null
      await tick()

      const config = await window.api.getAdapterConfig(adapter)
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
        selectedModel = node as ModelData
        // Base form data on the node, existing data is merged in loadAdapterConfig
        formData = { ...node }
        if (selectedModel.adapter) {
          loadAdapterConfig(selectedModel.adapter)
        } else {
          isLoading = false
          formFields = null
        }
      } else if (node.type === 'audio') {
        selectedModel = null
        formFields = null
        formData = { init_audio: node.name }
        isLoading = false // Not loading adapter config until a model is selected
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

  function handleModelSelect(newNode: ModelData): void {
    const newModel = newNode as ModelData
    const oldAdapter = selectedModel?.adapter
    selectedModel = newModel

    if (newModel.adapter !== oldAdapter) {
      // If the adapter is different, we need to load new config.
      // Keep existing form data where field names overlap.
      loadAdapterConfig(newModel.adapter)
    }
  }

  async function generate(): Promise<void> {
    const title = 'Generation'
    if (!selectedModel) {
      onError({ title: title + ' Failed', message: 'A model must be selected for generation.' })
      return
    }

    // Construct a clean payload, including only the model name and the fields
    // defined by the engine's configuration. This avoids sending the entire
    // (and potentially non-serializable) formData object.
    const payload: Record<string, unknown> = {
      model_id: selectedModel.id
    }

    if (formFields) {
      for (const field of formFields) {
        if (formData[field.name] !== undefined) {
          payload[field.name] = formData[field.name]
        }
      }
    }

    // Handle special cases like `init_audio` that are not part of the dynamic form
    if (formData.init_audio) {
      payload.init_audio = formData.init_audio
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
      <DynamicForm config={formFields} bind:formData bind:isFormValid />
    {:else if node.type === 'audio'}
      <p class="centered-text">Select a model to see its generation options.</p>
    {/if}
  </div>

  <div class="panel-actions">
    <button onclick={onClose}>Cancel</button>

    <button
      class="primary"
      onclick={generate}
      disabled={isLoading || !!error || !selectedModel || !isFormValid}
    >
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
