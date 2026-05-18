<script lang="ts">
  import { tick, onDestroy } from 'svelte'
  import type { FormConfig, NodeData } from '../../utils/forms'
  import { initializeFormData } from '../../utils/forms'
  import {
    initiatorNodeStore,
    contextStore,
    lastUsedModelStore,
    formStateStore
  } from '../../utils/stores'
  import { startExecution } from '../../utils/execution'
  import DynamicForm from '../DynamicForm.svelte'
  import NodeSelector from '../NodeSelector.svelte'
  import type { ErrorInfo } from '../../utils/types'

  let { onClose, onInvert, onError } = $props<{
    onClose: () => void
    onInvert: (data: unknown) => void
    onError: (error: ErrorInfo) => void
  }>()

  let adapterFields: FormConfig | null = $state(null)
  let formData: Record<string, unknown> = $state({})
  let isLoading = $state(true)
  let isInverting = $state(false)
  let error: string | null = $state(null)
  let isFormValid = $state(false)
  let contextData = $state({})
  let dynamicForm: DynamicForm | undefined = $state()
  let modelNodeSelector: ReturnType<typeof NodeSelector>

  function handleError(error: ErrorInfo): void {
    onError(error)
    onClose()
  }

  onDestroy(() => {
    formStateStore.clearGenerationModel()
  })

  $effect(() => {
    const data: Record<string, unknown> = {}

    if ($contextStore) {
      Object.assign(data, $contextStore)
    }

    if ($formStateStore.generationModel) {
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
      if (config && config.invert && Array.isArray(config.invert)) {
        adapterFields = config.invert
        const { formData: newFormData } = initializeFormData(
          adapterFields,
          $contextStore,
          $initiatorNodeStore
        )
        formData = newFormData
      } else if (config && config.generate && Array.isArray(config.generate)) {
        // Fallback to generation config just to scaffold the UI if invert doesn't exist yet
        adapterFields = config.generate
        const { formData: newFormData } = initializeFormData(
          adapterFields,
          $contextStore,
          $initiatorNodeStore
        )
        formData = newFormData
      } else {
        throw new Error('Invalid config format received from backend.')
      }
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : 'Failed to load configuration.'
      console.error('Failed to fetch form config:', e)
      error = message
      handleError({ title: 'Configuration Error', message })
    } finally {
      isLoading = false
    }
  }

  $effect(() => {
    const context = $contextStore

    if (context && context.model_id) {
      const modelId = context.model_id as string
      if (modelId && modelNodeSelector) {
        modelNodeSelector.selectNodeById(modelId)
      }
    } else if ($initiatorNodeStore || context) {
      $formStateStore.generationModel = $lastUsedModelStore
    } else {
      $formStateStore.generationModel = null
    }
  })

  $effect(() => {
    const model = $formStateStore.generationModel
    if (model) {
      if (model.adapter) {
        loadAdapterConfig(model.adapter)
      } else {
        isLoading = false
        adapterFields = null
      }
    } else {
      isLoading = false
      adapterFields = null
    }
  })

  function invert(): void {
    const title = 'Inversion'
    if (!$formStateStore.generationModel) {
      handleError({ title: title + ' Failed', message: 'A model must be selected for inversion.' })
      return
    }

    lastUsedModelStore.set($formStateStore.generationModel)

    const payload = dynamicForm.getPayload()
    const jobName = $formStateStore.generationModel.name || 'Inversion'

    const singlePayload = {
      ...payload,
      model_id: $formStateStore.generationModel.id,
      source_audio_id: $initiatorNodeStore?.id
    }
    
    console.log('Inverting with payload:', singlePayload)
    
    startExecution(jobName, singlePayload, 'invert').catch(console.error)
    onClose()
  }
</script>

<div class="view-container">
  <div class="view-content">
    <NodeSelector
      label="Model"
      filter={{ type: 'model' }}
      bind:node={$formStateStore.generationModel}
      bind:this={modelNodeSelector}
      id="model-selector"
    />

    {#if isLoading}
      <p>Loading configuration...</p>
    {:else if error}
      <div class="error-message">
        <p>Error: {error}</p>
      </div>
    {:else if isInverting}
      <p class="centered-text">Inverting...</p>
    {:else if adapterFields}
      <DynamicForm
        bind:this={dynamicForm}
        config={adapterFields}
        bind:formData
        bind:isFormValid
        {contextData}
      />
    {:else if !$formStateStore.generationModel}
      <p class="centered-text">Select a model to see its inversion options.</p>
    {/if}
  </div>

  <div class="panel-actions">
    <button onclick={onClose}>Cancel</button>

    <button
      class="primary"
      onclick={invert}
      disabled={isLoading ||
        !!error ||
        !$formStateStore.generationModel ||
        !isFormValid ||
        isInverting}
    >
      {#if isInverting}
        Inverting...
      {:else}
        Invert
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