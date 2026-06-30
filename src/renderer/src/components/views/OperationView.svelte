<!-- src/renderer/src/components/views/OperationView.svelte -->
<script lang="ts">
  import { tick, onDestroy, untrack } from 'svelte'
  import type { FormConfig, FormField, NodeData } from '../../utils/forms'
  import { initializeFormData } from '../../utils/forms'
  import {
    initiatorNodeStore,
    selectedOperation,
    contextStore,
    cyInstanceStore
  } from '../../utils/stores'
  import { startExecution } from '../../utils/execution'
  import DynamicForm from '../DynamicForm.svelte'
  import NodeSelectorList, { type NodeListItem } from '../NodeSelectorList.svelte'
  import type { ErrorInfo } from '../../utils/types'

  let { onClose, onError } = $props<{
    onClose: () => void
    onError: (error: ErrorInfo) => void
  }>()

  let baseFields = $derived(($selectedOperation?.form_config || []) as FormConfig)
  let adapterFields = $state<FormField[]>([])
  let fieldsConfig = $derived([...baseFields, ...adapterFields])

  let formData: Record<string, unknown> = $state({})
  let isLoading = $state(true)
  let isRunning = $state(false)
  let isFormValid = $state(false)
  let contextData = $state({})
  let dynamicForm: DynamicForm | undefined = $state()
  let error: string | null = $state(null)
  let lastLoadedModelId: string | null = $state(null)

  // Gratings sub-selection state (for generation operation)
  let selectedGratings: NodeListItem[] = $state([])

  let parentBatchId = $derived.by(() => {
    const cy = $cyInstanceStore
    const node = $initiatorNodeStore
    if (cy && node && node.parent) {
      const parentNode = cy.$id(node.parent)
      if (parentNode && parentNode.length > 0 && parentNode.data('type') === 'batch') {
        return node.parent
      }
    }
    return null
  })

  let isReplicated = $derived(!!$contextStore)
  let addToSameBatch = $state(true)

  onDestroy(() => {
    // Clear state on destroy
    selectedOperation.set(null)
  })

  // Keep contextData updated
  $effect(() => {
    const data: Record<string, unknown> = {}
    if ($contextStore) {
      Object.assign(data, $contextStore)
    }
    if ($initiatorNodeStore) {
      data.initiator = $initiatorNodeStore
    }
    contextData = data
  })

  // Watch for selected operation changes and initialize form data
  $effect(() => {
    const op = $selectedOperation
    if (op) {
      isLoading = true
      error = null
      adapterFields = []
      selectedGratings = []
      lastLoadedModelId = null
      addToSameBatch = true

      const baseFieldsConfig = (op.form_config || []) as FormConfig

      const { formData: initialData } = initializeFormData(
        baseFieldsConfig,
        $contextStore,
        isReplicated ? null : $initiatorNodeStore
      )

      // Pre-population for node fields
      if (op.execution_mode === 'async') {
        let modelId: string | null = null
        if ($contextStore?.model_id) {
          modelId = $contextStore.model_id
        } else if ($initiatorNodeStore?.base_model_id) {
          modelId = $initiatorNodeStore.base_model_id
        } else if ($initiatorNodeStore?.context?.model_id) {
          modelId = $initiatorNodeStore.context.model_id
        } else if ($initiatorNodeStore?.type === 'model') {
          modelId = $initiatorNodeStore.id
        }

        if (modelId) {
          initialData.model = modelId
        }

        if (isReplicated && $contextStore?.gratings && $cyInstanceStore) {
          const contextGratings = $contextStore.gratings as Array<{ id: string; strength: number }>
          selectedGratings = contextGratings.map((g, index) => {
            const node = $cyInstanceStore.$id(g.id).data()
            return {
              id: index,
              node: node || g.id,
              strength: g.strength
            }
          })
        }

        if ($initiatorNodeStore && !isReplicated) {
          const type = $initiatorNodeStore.type
          const effectiveType = type === 'batch' ? $initiatorNodeStore.member_type : type
          if (effectiveType === 'audio') {
            if (baseFieldsConfig.some((f) => f.name === 'init_audio')) {
              initialData.init_audio = $initiatorNodeStore
            }
            if (baseFieldsConfig.some((f) => f.name === 'source_audio')) {
              initialData.source_audio = $initiatorNodeStore
            }
          } else if (effectiveType === 'latent') {
            if (baseFieldsConfig.some((f) => f.name === 'init_latent')) {
              initialData.init_latent = $initiatorNodeStore
            }
          } else if (effectiveType === 'grating') {
            selectedGratings = [
              { id: -1, node: $initiatorNodeStore as unknown as NodeData, strength: 1.0 }
            ]
          }
        }
      } else {
        // Synchronous operations fallback
        if (!isReplicated) {
          for (const field of baseFieldsConfig) {
            if (field.type === 'node') {
              const isAudioInitiator =
                $initiatorNodeStore?.type === 'audio' ||
                ($initiatorNodeStore?.type === 'batch' &&
                  $initiatorNodeStore?.member_type === 'audio')
              if (field.name === 'source_audio' && isAudioInitiator) {
                initialData[field.name] = $initiatorNodeStore
              }
            }
          }
        }
      }

      formData = initialData
      isLoading = false
    } else {
      isLoading = false
      formData = {}
    }
  })

  // Load dynamic adapter fields when model is chosen in async operations
  $effect(() => {
    const op = $selectedOperation
    const modelNode = formData.model
    const cy = $cyInstanceStore

    untrack(() => {
      if (op && op.execution_mode === 'async' && modelNode) {
        const isObj = typeof modelNode === 'object' && modelNode !== null
        const rawModelId = isObj
          ? String((modelNode as Record<string, unknown>).id || '')
          : String(modelNode)

        // Extract the first model ID if it is a comma-separated list
        const modelId = rawModelId.split(',')[0].trim()

        let adapter = isObj
          ? ((modelNode as Record<string, unknown>).adapter as string | null)
          : null
        let config = isObj
          ? ((modelNode as Record<string, unknown>).config as Record<string, any> | null)
          : null

        if (cy && modelId) {
          const cyNode = cy.$id(modelId)
          if (cyNode && cyNode.length > 0) {
            if (!adapter) {
              adapter = cyNode.data('adapter') as string | null
            }
            if (!config) {
              config = cyNode.data('config') as Record<string, any> | null
            }
          }
        }

        let modelType: string | null = null
        let diffusionObjective: string = 'v'
        if (config) {
          const modelConfig = config.model || {}
          const diffusionConfig = modelConfig.diffusion || {}
          diffusionObjective = diffusionConfig.diffusion_objective || 'v'
          modelType = ['rectified_flow', 'rf_denoiser'].includes(diffusionObjective)
            ? 'rectified_flow'
            : 'k_diffusion'
        }

        // Expose model_type at the top level of formData for show_if visibility rules
        if (modelType && formData.model_type !== modelType) {
          formData.model_type = modelType
          formData.sampler_category = modelType
        }
        if (diffusionObjective && formData.diffusion_objective !== diffusionObjective) {
          formData.diffusion_objective = diffusionObjective
        }

        if (adapter) {
          if (modelId !== lastLoadedModelId) {
            lastLoadedModelId = modelId
            loadAdapterFields(adapter, op.name)
          }
        } else {
          adapterFields = []
          lastLoadedModelId = null
          formData.model_type = null
        }
      } else {
        adapterFields = []
        lastLoadedModelId = null
      }
    })
  })

  async function loadAdapterFields(adapterName: string, opName: string): Promise<void> {
    try {
      isLoading = true
      error = null
      await tick()

      const config = await window.api.getAdapterConfig(adapterName)
      const opFields = config?.[opName]
      if (opFields && Array.isArray(opFields)) {
        adapterFields = opFields

        const { formData: adapterData } = initializeFormData(
          opFields as FormConfig,
          $contextStore,
          isReplicated ? null : $initiatorNodeStore
        )

        // Merge fields while keeping existing formData keys intact
        formData = { ...adapterData, ...formData }
      } else {
        throw new Error(`Adapter "${adapterName}" does not support operation "${opName}".`)
      }
    } catch (e: unknown) {
      console.error('Failed to load adapter fields:', e)
      error = e instanceof Error ? e.message : String(e)
      lastLoadedModelId = null
    } finally {
      isLoading = false
    }
  }

  function parseSequence(str: string): (string | number)[] {
    const parts = str.split(',').map((p) => p.trim())
    const result: (string | number)[] = []

    for (const part of parts) {
      const rangeMatch = part.match(/^(-?\d+\.?\d*)-(-?\d+\.?\d*):(-?\d+\.?\d*)$/)
      if (rangeMatch) {
        const [, start, end, step] = rangeMatch.map(Number)
        for (let i = start; i <= end; i += step) {
          result.push(i)
        }
        continue
      }

      const simpleRangeMatch = part.match(/^(-?\d+\.?\d*)-(-?\d+\.?\d*)$/)
      if (simpleRangeMatch) {
        const [, start, end] = simpleRangeMatch.map(Number)
        for (let i = start; i <= end; i++) {
          result.push(i)
        }
        continue
      }

      if (!isNaN(Number(part))) {
        result.push(Number(part))
        continue
      }

      result.push(part)
    }

    return result
  }

  function cartesian<T>(...arrays: T[][]): T[][] {
    return arrays.reduce((a, b) => a.flatMap((x) => b.map((y) => [...x, y])), [[]] as T[][])
  }

  function execute(): void {
    const op = $selectedOperation
    if (!op) return

    const payload = dynamicForm ? dynamicForm.getPayload() : {}
    const batchFields = dynamicForm ? dynamicForm.getBatchFields() : new Set<string>()
    const jobName = op.name ? op.name.toUpperCase() : 'Operation'

    const basePayload: Record<string, unknown> = { ...payload }

    // Map Gratings if the operation supports them
    if (op.name === 'generate') {
      const validGratings = selectedGratings.filter((l) => l.node)
      const gratings = validGratings.map((l) => ({
        id: typeof l.node === 'string' ? l.node : l.node?.id,
        strength: l.strength ?? 1.0
      }))
      if (gratings.length > 0) {
        basePayload.gratings = gratings
      }
    }

    if (batchFields.size === 0) {
      // Add fallback properties for backend compatibility
      if (basePayload.source_audio && !basePayload.source_audio_id) {
        basePayload.source_audio_id = basePayload.source_audio
      }
      if (basePayload.model && !basePayload.model_id) {
        basePayload.model_id = basePayload.model
      }

      if (parentBatchId && isReplicated && addToSameBatch) {
        basePayload.batch_id = parentBatchId
      }

      // Single run
      runJob(jobName, basePayload, op.name, op.execution_mode)
      onClose()
      return
    }

    // Batch run (Cartesian Product)
    isRunning = true
    try {
      const batchParams: Record<string, (string | number)[]> = {}
      const staticParams: Record<string, unknown> = {}

      for (const key in basePayload) {
        if (batchFields.has(key)) {
          batchParams[key] = parseSequence(String(basePayload[key]))
        } else {
          staticParams[key] = basePayload[key]
        }
      }

      const paramNames = Object.keys(batchParams)
      const paramValues = Object.values(batchParams)
      const combinations = cartesian(...paramValues)

      console.log(`Starting batch execution with ${combinations.length} combinations.`)

      const batchId =
        parentBatchId && isReplicated && addToSameBatch
          ? parentBatchId
          : `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`

      combinations.forEach((combination) => {
        const batchPayload: Record<string, unknown> = { ...staticParams }
        combination.forEach((value, index) => {
          batchPayload[paramNames[index]] = value
        })

        // Add fallback properties for backend compatibility
        if (batchPayload.source_audio && !batchPayload.source_audio_id) {
          batchPayload.source_audio_id = batchPayload.source_audio
        }
        if (batchPayload.model && !batchPayload.model_id) {
          batchPayload.model_id = batchPayload.model
        }

        batchPayload.batch_id = batchId

        runJob(jobName, batchPayload, op.name, op.execution_mode)
      })

      onClose()
    } catch (e: unknown) {
      console.error('Batch execution failed:', e)
      onError({
        title: 'Batch Failed',
        message: e instanceof Error ? e.message : String(e)
      })
    } finally {
      isRunning = false
    }
  }

  function runJob(jobName: string, payload: unknown, opName: string, mode: 'sync' | 'async'): void {
    startExecution(jobName, payload, opName, mode).catch((err: unknown) => {
      console.error('Execution failed:', err)
      onError({
        title: 'Execution Failed',
        message: err instanceof Error ? err.message : String(err)
      })
    })
  }
</script>

<div class="view-container">
  <div class="view-content">
    {#if isLoading}
      <p class="centered-text">Loading configuration...</p>
    {:else if error}
      <div class="error-message">
        <p>Error: {error}</p>
      </div>
    {:else if $selectedOperation}
      <p class="op-description">{$selectedOperation.description || 'No description available.'}</p>

      {#if fieldsConfig && fieldsConfig.length > 0}
        <DynamicForm
          bind:this={dynamicForm}
          config={fieldsConfig}
          bind:formData
          bind:isFormValid
          {contextData}
        />
      {/if}

      <!-- Render optional gratings if operation is generate and model is set -->
      {#if $selectedOperation.name === 'generate' && formData.model}
        <div style="margin-top: 1.5rem;">
          <NodeSelectorList
            title="Gratings (Optional)"
            addButtonText="Add Grating"
            filter={{
              type: 'grating',
              base_model_id:
                typeof formData.model === 'object' && formData.model
                  ? String((formData.model as Record<string, unknown>).id || '')
                  : String(formData.model)
            }}
            bind:items={selectedGratings}
            idPrefix="grating"
            showStrengths={true}
          />
        </div>
      {/if}

      {#if parentBatchId && isReplicated}
        <div class="options">
          <label>
            <input type="checkbox" bind:checked={addToSameBatch} disabled={isRunning} />
            Add new artifact to the same batch
          </label>
        </div>
      {/if}

      {#if (!fieldsConfig || fieldsConfig.length === 0) && !(parentBatchId && isReplicated)}
        <p class="centered-text">No parameters needed for this operation.</p>
      {/if}
    {:else}
      <p class="centered-text">No operation selected.</p>
    {/if}
  </div>

  <div class="panel-actions">
    <button onclick={onClose} disabled={isRunning}>Cancel</button>

    <button
      class="primary"
      onclick={execute}
      disabled={isLoading ||
        !$selectedOperation ||
        (fieldsConfig?.length > 0 && !isFormValid) ||
        isRunning}
    >
      {#if isRunning}
        Running...
      {:else}
        Run
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
  .op-description {
    color: var(--color-text-muted, #aaa);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
    line-height: 1.4;
  }
  .centered-text {
    text-align: center;
    padding: 2rem 1rem;
    color: var(--color-text-muted, #aaa);
  }
  .error-message {
    padding: 1rem;
    color: var(--color-error, #ef4444);
    background: rgba(239, 68, 68, 0.1);
    border-radius: 0.25rem;
    margin-bottom: 1rem;
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
  .options {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--color-border-glass-1);
  }
  .options label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    font-size: 0.9rem;
    color: var(--color-overlay-text);
  }
  .options input[type='checkbox'] {
    cursor: pointer;
    width: auto;
    margin: 0;
  }
</style>
