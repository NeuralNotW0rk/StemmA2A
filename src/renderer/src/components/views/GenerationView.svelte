<script lang="ts">
  import { tick } from 'svelte'
  import type { FormConfig, ModelData, AudioData } from '../../utils/forms'
  import { activeNodeStore, lastUsedModelStore } from '../../utils/stores'
  import DynamicForm from '../DynamicForm.svelte'
  import NodeSelector from '../NodeSelector.svelte'

  type NodeData = ModelData | AudioData

  let { node, onClose, onGenerate, onError } = $props<{
    node: NodeData
    onClose: () => void
    onGenerate: (data: unknown) => void
    onError: (error: { title: string; message: string }) => void
  }>()

  let adapterFields: FormConfig | null = $state(null)
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
      adapterFields = null
      await tick()

      const config = await window.api.getAdapterConfig(adapter)
      if (config && config.generate && Array.isArray(config.generate)) {
        adapterFields = config.generate
        const newFormData = {}
        for (const field of adapterFields) {
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
      error = null
      inProgress = false
      formData = {}

      if (node.type === 'model') {
        selectedModel = node as ModelData
      } else if (node.type === 'audio') {
        // If an audio node is opened, use the last used model
        selectedModel = $lastUsedModelStore
        // And set this audio as the init_audio in the form, if the form has such a field
        if (adapterFields?.some((f) => f.name === 'init_audio')) {
          formData.init_audio = node
        }
      }
    }
  })

  $effect(() => {
    if (selectedModel) {
      activeNodeStore.set(selectedModel)
      if (selectedModel.adapter) {
        loadAdapterConfig(selectedModel.adapter)
      } else {
        isLoading = false
        adapterFields = null
      }
    } else {
      activeNodeStore.set(node)
      isLoading = false
      adapterFields = null
    }
  })

  function handleModelSelect(newModel: ModelData): void {
    const oldAdapter = selectedModel?.adapter
    selectedModel = newModel

    if (newModel.adapter !== oldAdapter) {
      loadAdapterConfig(newModel.adapter)
    }
  }

  async function generate(): Promise<void> {
    const title = 'Generation'
    if (!selectedModel) {
      onError({ title: title + ' Failed', message: 'A model must be selected for generation.' })
      return
    }

    lastUsedModelStore.set(selectedModel)

    const payload: Record<string, unknown> = { model_id: selectedModel.id }
    if (adapterFields) {
      for (const field of adapterFields) {
        const fieldName = field.name
        const value = formData[fieldName]

        if (value !== undefined && value !== null) {
          if (field.type === 'node' && typeof value === 'object' && 'id' in value) {
            payload[`${fieldName}_id`] = (value as { id: string }).id
          } else {
            payload[fieldName] = value
          }
        }
      }
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
    {:else if adapterFields}
      <DynamicForm
        config={adapterFields}
        bind:formData
        bind:isFormValid
        contextData={selectedModel}
      />
    {:else if !selectedModel}
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
