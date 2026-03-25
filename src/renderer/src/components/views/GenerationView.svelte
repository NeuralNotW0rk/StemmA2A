<script lang="ts">
  import { tick, onDestroy } from 'svelte'
  import type { FormConfig, ModelData, NodeData } from '../../utils/forms'
  import { initializeFormData } from '../../utils/forms'
  import {
    initiatorNodeStore,
    boundNodeStore,
    lastUsedModelStore,
    formStateStore
  } from '../../utils/stores'
  import { startExecution } from '../../utils/execution'
  import DynamicForm from '../DynamicForm.svelte'
  import NodeSelector from '../NodeSelector.svelte'

  let { onClose, onError } = $props<{
    onClose: () => void
    onGenerate: (data: unknown) => void
    onError: (error: { title: string; message: string }) => void
  }>()

  let adapterFields: FormConfig | null = $state(null)
  let formData: Record<string, unknown> = $state({})
  let isLoading = $state(true)
  let error: string | null = $state(null)
  let isFormValid = $state(false)
  let contextData = $state({})
  let dynamicForm: DynamicForm | undefined = $state()

  // Clear the store when the component is destroyed to prevent state leakage.
  onDestroy(() => {
    formStateStore.clearGenerationModel()
  })

  $effect(() => {
    const data: Record<string, unknown> = {}
    if ($formStateStore.generationModel) {
      // Flatten model properties into the context for show_if conditions
      Object.assign(data, $formStateStore.generationModel)
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
    // This effect runs when the component mounts. It sets the initial state
    // of the form's model in the global store.
    const initiatorNode = $initiatorNodeStore
    if (initiatorNode) {
      if (initiatorNode.type === 'model') {
        $formStateStore.generationModel = initiatorNode as ModelData
      } else if (initiatorNode.type === 'audio') {
        // If an audio node is opened, use the last used model
        $formStateStore.generationModel = $lastUsedModelStore
      }
    } else {
      $formStateStore.generationModel = null
    }
  })

  $effect(() => {
    // This effect runs whenever the model in the store changes.
    const model = $formStateStore.generationModel
    if (model) {
      if (model.adapter) {
        loadAdapterConfig(model.adapter)
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
    if ($formStateStore.generationModel) {
      newBoundNodes['model'] = $formStateStore.generationModel
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

  function generate(): void {
    const title = 'Generation'
    if (!$formStateStore.generationModel) {
      onError({ title: title + ' Failed', message: 'A model must be selected for generation.' })
      return
    }

    lastUsedModelStore.set($formStateStore.generationModel)

    const payload = {
      ...dynamicForm.getPayload(),
      model_id: $formStateStore.generationModel.id
    }

    console.log('Generating with payload:', payload)
    startExecution(payload)
    onClose()
  }
</script>

<div class="view-container">
  <div class="view-content">
    <NodeSelector
      label="Model"
      selectionType="model"
      bind:node={$formStateStore.generationModel}
      id="model-selector"
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
    {:else if !$formStateStore.generationModel}
      <p class="centered-text">Select a model to see its generation options.</p>
    {/if}
  </div>

  <div class="panel-actions">
    <button onclick={onClose}>Cancel</button>

    <button
      class="primary"
      onclick={generate}
      disabled={isLoading || !!error || !$formStateStore.generationModel || !isFormValid}
    >
      Generate
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
