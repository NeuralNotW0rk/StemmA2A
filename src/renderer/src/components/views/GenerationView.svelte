<script lang="ts">
  import { tick } from 'svelte'
  import type { FormConfig, ModelData, NodeData } from '../../utils/forms'
  import { initializeFormData } from '../../utils/forms'
  import { initiatorNodeStore, boundNodeStore, lastUsedModelStore } from '../../utils/stores'
  import DynamicForm from '../DynamicForm.svelte'
  import NodeSelector from '../NodeSelector.svelte'

  let { onClose, onGenerate, onError } = $props<{
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
  let isFormValid = $state(false)
  let contextData = $state({})
  let dynamicForm: DynamicForm | undefined = $state()

  $effect(() => {
    const data: Record<string, unknown> = {}
    if (selectedModel) {
      // Flatten model properties into the context for show_if conditions
      Object.assign(data, selectedModel)
    }

    if ($initiatorNodeStore) {
      data.initiator = $initiatorNodeStore
    }

    contextData = data
  })

  async function loadAdapterConfig(adapter: string): Promise<void> {
    try {
      isLoading = true
      error = null
      adapterFields = null
      await tick()

      const config = await window.api.getAdapterConfig(adapter)
      if (config && config.generate && Array.isArray(config.generate)) {
        adapterFields = config.generate
        const { formData: newFormData, boundNodes: newBoundNodes } = initializeFormData(
          adapterFields,
          $initiatorNodeStore
        )
        formData = newFormData
        boundNodeStore.set(newBoundNodes)
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
    // This effect runs when the component mounts and whenever initiatorNodeStore changes.
    const initiatorNode = $initiatorNodeStore
    if (initiatorNode) {
      error = null
      inProgress = false
      formData = {}

      if (initiatorNode.type === 'model') {
        selectedModel = initiatorNode as ModelData
      } else if (initiatorNode.type === 'audio') {
        // If an audio node is opened, use the last used model
        selectedModel = $lastUsedModelStore
      }
    } else {
      // Reset if there's no initiator node
      selectedModel = null
    }
  })

  $effect(() => {
    // This effect runs whenever selectedModel changes, handling async loading.
    if (selectedModel) {
      if (selectedModel.adapter) {
        loadAdapterConfig(selectedModel.adapter)
      } else {
        isLoading = false
        adapterFields = null
      }
    } else {
      // If no model is selected, we are not loading anything.
      isLoading = false
      adapterFields = null
    }
  })

  $effect(() => {
    // This effect is the single source of truth for what is 'bound' in the graph.
    const newBoundNodes = {}

    // The initiator node is always bound.
    if ($initiatorNodeStore) {
      newBoundNodes['initiator'] = $initiatorNodeStore
    }

    // The selected model is also bound.
    if (selectedModel) {
      newBoundNodes['model'] = selectedModel
    }

    // Any nodes selected in the form are also bound.
    if (adapterFields) {
      for (const field of adapterFields) {
        if (field.type === 'node') {
          const node = formData[field.name] as NodeData | null
          if (node) {
            newBoundNodes[field.name] = node
          }
        }
      }
    }

    boundNodeStore.set(newBoundNodes)
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

    const payload = {
      ...dynamicForm.getPayload(),
      model_id: selectedModel.id
    }

    console.log('Generating with payload:', payload)
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
        bind:this={dynamicForm}
        config={adapterFields}
        bind:formData
        bind:isFormValid
        {contextData}
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
