<script lang="ts">
  import { tick, onDestroy } from 'svelte'
  import type { FormConfig } from '../../utils/forms'
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

  let { onClose, onError } = $props<{
    onClose: () => void
    onGenerate: (data: unknown) => void
    onError: (error: { title: string; message: string }) => void
  }>()

  let adapterFields: FormConfig | null = $state(null)
  let formData: Record<string, unknown> = $state({})
  let isLoading = $state(true)
  let isGenerating = $state(false)
  let error: string | null = $state(null)
  let isFormValid = $state(false)
  let contextData = $state({})
  let dynamicForm: DynamicForm | undefined = $state()
  let modelNodeSelector: ReturnType<typeof NodeSelector>

  function handleError(error: { title: string; message: string }): void {
    onError(error)
    onClose()
  }

  // Clear the store when the component is destroyed to prevent state leakage.
  onDestroy(() => {
    formStateStore.clearGenerationModel()
  })

  $effect(() => {
    const data: Record<string, unknown> = {}

    if ($contextStore) {
      Object.assign(data, $contextStore)
    }

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
        const isReplication = !!$contextStore && $initiatorNodeStore?.type === 'audio'
        const { formData: newFormData } = initializeFormData(
          adapterFields,
          $contextStore,
          isReplication ? null : $initiatorNodeStore
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
    // This effect runs when the component mounts. It sets the initial state
    // of the form's model in the global store.
    const context = $contextStore

    if (context && context.model_id) {
      const modelId = context.model_id as string
      if (modelId && modelNodeSelector) {
        modelNodeSelector.selectNodeById(modelId)
      }
    } else if ($initiatorNodeStore || context) {
      // If opened without a specific model_id, use the last used model
      $formStateStore.generationModel = $lastUsedModelStore
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

  function parseSequence(str: string): (string | number)[] {
    const parts = str.split(',').map((p) => p.trim())
    const result: (string | number)[] = []

    for (const part of parts) {
      // Range with step, e.g., "1-10:2"
      const rangeMatch = part.match(/^(-?\d+\.?\d*)-(-?\d+\.?\d*):(-?\d+\.?\d*)$/)
      if (rangeMatch) {
        const [, start, end, step] = rangeMatch.map(Number)
        for (let i = start; i <= end; i += step) {
          result.push(i)
        }
        continue
      }

      // Range without step, e.g., "1-5"
      const simpleRangeMatch = part.match(/^(-?\d+\.?\d*)-(-?\d+\.?\d*)$/)
      if (simpleRangeMatch) {
        const [, start, end] = simpleRangeMatch.map(Number)
        for (let i = start; i <= end; i++) {
          result.push(i)
        }
        continue
      }

      // Single number
      if (!isNaN(Number(part))) {
        result.push(Number(part))
        continue
      }

      // String value
      result.push(part)
    }

    return result
  }

  function cartesian<T>(...arrays: T[][]): T[][] {
    return arrays.reduce((a, b) => a.flatMap((x) => b.map((y) => [...x, y])), [[]] as T[][])
  }

  function generate(): void {
    const title = 'Generation'
    if (!$formStateStore.generationModel) {
      handleError({ title: title + ' Failed', message: 'A model must be selected for generation.' })
      return
    }

    lastUsedModelStore.set($formStateStore.generationModel)

    const payload = dynamicForm.getPayload()
    const batchFields = dynamicForm.getBatchFields()
    const jobName = $formStateStore.generationModel.name || 'Generation'

    if (batchFields.size === 0) {
      // Single generation
      const singlePayload = {
        ...payload,
        model_id: $formStateStore.generationModel.id
      }
      console.log('Generating with payload:', singlePayload)
      startExecution(jobName, singlePayload)
      onClose()
      return
    }

    // Batch generation
    isGenerating = true
    const batchParams: Record<string, (string | number)[]> = {}
    const staticParams: Record<string, unknown> = {}

    for (const key in payload) {
      if (batchFields.has(key)) {
        batchParams[key] = parseSequence(String(payload[key]))
      } else {
        staticParams[key] = payload[key]
      }
    }

    const paramNames = Object.keys(batchParams)
    const paramValues = Object.values(batchParams)
    const combinations = cartesian(...paramValues)

    console.log(`Starting batch generation with ${combinations.length} combinations.`)

    // Create a unique ID for this batch operation
    const batchId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`

    for (const combination of combinations) {
      const batchPayload: Record<string, unknown> = { ...staticParams }
      combination.forEach((value, index) => {
        batchPayload[paramNames[index]] = value
      })

      const fullPayload = {
        ...batchPayload,
        model_id: $formStateStore.generationModel.id,
        batch_id: batchId
      }
      console.log('Generating with payload:', fullPayload)
      startExecution(jobName, fullPayload)
    }

    isGenerating = false
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
    {:else if isGenerating}
      <p class="centered-text">Generating in batch...</p>
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
      disabled={isLoading ||
        !!error ||
        !$formStateStore.generationModel ||
        !isFormValid ||
        isGenerating}
    >
      {#if isGenerating}
        Generating...
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
