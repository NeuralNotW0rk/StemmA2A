<!-- src/renderer/src/components/views/InitializeEvolutionView.svelte -->
<script lang="ts">
  import { onMount } from 'svelte'
  import NodeSelector from '../NodeSelector.svelte'
  import type { NodeData } from '../../utils/forms'
  import type { ErrorInfo } from '../../utils/types'
  import { cyInstanceStore } from '../../utils/stores'
  import { jobStore, pollJobStatus, updateJob, type JobStatus } from '../../utils/job-management'

  interface ModelLayer {
    address: string
    type: string
    shape?: number[]
    in_features?: number
    out_features?: number
  }

  let { audioElement, onclose, onrefresh, onError } = $props<{
    audioElement: NodeData | null
    onclose: () => void
    onrefresh: () => void
    onError: (error: ErrorInfo) => void
  }>()

  // State
  let selectedModel = $state<NodeData | null>(null)
  let loraRank = $state(4)
  let loraAlpha = $state(1.0)
  let populationSize = $state(10)
  let loraNoise = $state(0.05)
  let activeFlipProb = $state(0.05)

  let inProgress = $state(false)
  let layers = $state<ModelLayer[]>([])
  let loadingLayers = $state(true)

  // Try to resolve the model ID from cytoscape or context
  $effect((): void => {
    if (audioElement && $cyInstanceStore && !selectedModel) {
      const context = (audioElement as any).context || {}
      const modelId = context.model_id

      if (modelId) {
        const modelNode = $cyInstanceStore.$id(modelId).data()
        if (modelNode) {
          selectedModel = modelNode as NodeData
        } else {
          selectedModel = { id: modelId, name: modelId, type: 'model', adapter: '' }
        }
      } else {
        const cyNode = $cyInstanceStore.$id(audioElement.id)
        if (cyNode.length > 0) {
          const modelIncomer = cyNode.incomers('node[type="model"]')
          if (modelIncomer.length > 0) {
            selectedModel = modelIncomer.data() as NodeData
          }
        }
      }
    }
  })

  // Load layers when selected model changes
  $effect((): (() => void) | void => {
    const modelId = selectedModel?.id
    if (!modelId) {
      layers = []
      loadingLayers = false
      return
    }

    let isCurrent = true
    loadingLayers = true

    window.api
      .getModelLayers(modelId)
      .then((response: unknown): void => {
        if (!isCurrent) return

        if (
          response &&
          typeof response === 'object' &&
          'success' in response &&
          (response as Record<string, unknown>).success === true &&
          'layers' in response &&
          Array.isArray((response as Record<string, unknown>).layers)
        ) {
          layers = (response as Record<string, unknown>).layers as ModelLayer[]
        } else {
          layers = []
        }
      })
      .catch((e: unknown): void => {
        if (!isCurrent) return
        const message = e instanceof Error ? e.message : String(e)
        console.error('Failed to load model layers:', message)
        layers = []
      })
      .finally((): void => {
        if (isCurrent) {
          loadingLayers = false
        }
      })

    return (): void => {
      isCurrent = false
    }
  })

  // Filter for Linear layers which are compatible with LoRA
  let linearLayers = $derived(layers.filter((l): boolean => l.type.includes('Linear')))

  let isFormValid = $derived(
    !!selectedModel &&
      !loadingLayers &&
      linearLayers.length > 0 &&
      loraRank > 0 &&
      loraAlpha > 0 &&
      populationSize >= 2
  )

  async function startEvolutionInitialization(): Promise<void> {
    if (!isFormValid || !selectedModel) return

    inProgress = true
    try {
      // Construct the LoRA elements payload for all Linear layers
      const elements = linearLayers.map((l) => ({
        address: l.address,
        kernel_type: 'lora',
        params: {
          rank: loraRank,
          alpha: loraAlpha,
          in_features: l.in_features ?? 0,
          out_features: l.out_features ?? 0,
          kernel_size: null
        },
        indices: [],
        perform_clustering: false,
        num_clusters: null,
        cluster: null
      }))

      // 1. Build the generation context for evolved variants
      const generationContext = {
        ...((audioElement?.context as Record<string, unknown>) || {}),
        model_id: selectedModel.id
      }

      const evolutionPayload = {
        model_id: selectedModel.id,
        elements: elements,
        precursor_audio_id: audioElement.id,
        population_size: populationSize,
        lora_noise: loraNoise,
        active_flip_prob: activeFlipProb,
        generation_context: generationContext
      }

      // 2. Start evolution step to produce genomes and queue jobs (parent job)
      const evolutionResult = (await window.api.startEvolution(evolutionPayload)) as Record<string, unknown>
      if (
        !evolutionResult ||
        typeof evolutionResult !== 'object' ||
        evolutionResult.success !== true ||
        typeof evolutionResult.job_id !== 'string'
      ) {
        throw new Error('Backend failed to start evolution.')
      }

      const parentJobId = evolutionResult.job_id as string
      const parentJob = {
        id: parentJobId,
        name: 'Initializing Evolution Population',
        status: 'pending' as JobStatus,
        payload: evolutionPayload,
        progress: null,
        result: null,
        error: null,
        createdAt: Date.now(),
        updatedAt: Date.now()
      }

      jobStore.update((jobs) => [...jobs, parentJob])

      // Poll the local parent job status
      pollJobStatus(parentJobId)
        .then((res: unknown): void => {
          parentJob.status = 'success'
          parentJob.result = res as Record<string, unknown>
          updateJob(parentJob)

          // Upon success, register and poll each child job
          if (
            res &&
            typeof res === 'object' &&
            'result' in res &&
            res.result &&
            typeof res.result === 'object'
          ) {
            const resultData = res.result as Record<string, unknown>
            const jobIds = resultData.job_ids as string[]

            for (let i = 0; i < jobIds.length; i++) {
              const jId = jobIds[i]
              const childJob = {
                id: jId,
                name: `Evolution - Ind ${i + 1}`,
                status: 'pending' as JobStatus,
                payload: evolutionPayload,
                progress: null,
                result: null,
                error: null,
                createdAt: Date.now(),
                updatedAt: Date.now()
              }

              jobStore.update((jobs) => [...jobs, childJob])

              pollJobStatus(jId)
                .then((childRes: unknown): void => {
                  childJob.status = 'success'
                  childJob.result = childRes as Record<string, unknown>
                  updateJob(childJob)
                })
                .catch((childErr: unknown): void => {
                  childJob.status = 'error'
                  childJob.error = {
                    title: 'Evolution Job Failed',
                    message: childErr instanceof Error ? childErr.message : String(childErr)
                  }
                  updateJob(childJob)
                })
            }
          }
          onrefresh()
        })
        .catch((err: unknown): void => {
          parentJob.status = 'error'
          parentJob.error = {
            title: 'Evolution Initialization Failed',
            message: err instanceof Error ? err.message : String(err)
          }
          updateJob(parentJob)
        })

      onclose()
      onrefresh()
    } catch (e: unknown) {
      console.error('Failed to initialize evolution baseline grating:', e)
      const message = e instanceof Error ? e.message : String(e)
      onError({ title: 'Evolution Initialization Failed', message })
    } finally {
      inProgress = false
    }
  }
</script>

<div class="view-container">
  <div class="view-content">
    <div class="alert-box info">
      Evolving artifact: <strong
        >{audioElement?.name || audioElement?.alias || 'Selected Audio'}</strong
      >. This will create a new zero-initialized LoRA grating that matches the target model's
      structure.
    </div>

    <NodeSelector
      label="Target Model"
      filter={{ type: 'model' }}
      bind:node={selectedModel}
      id="target-model-selector"
    />

    <div class="form-row">
      <label>
        LoRA Rank
        <input type="number" min="1" max="128" bind:value={loraRank} />
      </label>
      <label>
        LoRA Alpha
        <input type="number" min="0.1" step="0.1" bind:value={loraAlpha} />
      </label>
    </div>

    <label>
      Population Size ($N$)
      <input type="number" min="2" max="100" bind:value={populationSize} />
    </label>

    <div class="noise-sliders">
      <span class="field-label">Initial Population Parameters</span>

      <label class="slider-label">
        <div class="slider-header">
          <span>LoRA Noise</span>
          <span>{loraNoise.toFixed(2)}</span>
        </div>
        <input type="range" min="0.0" max="2.0" step="0.01" bind:value={loraNoise} />
      </label>

      <label class="slider-label">
        <div class="slider-header">
          <span>Active Flip Probability</span>
          <span>{activeFlipProb.toFixed(2)}</span>
        </div>
        <input type="range" min="0.0" max="1.0" step="0.01" bind:value={activeFlipProb} />
      </label>
    </div>

    <div class="layer-status">
      <span class="field-label">Target Model Layers</span>
      {#if loadingLayers}
        <div class="loading-layers-box">
          <div class="spinner tiny"></div>
          <span>Inspecting model structure...</span>
        </div>
      {:else if linearLayers.length > 0}
        <div class="status-msg success">
          Found {linearLayers.length} compatible Linear layers for LoRA.
        </div>
      {:else}
        <div class="status-msg error">No compatible Linear layers found on this model.</div>
      {/if}
    </div>
  </div>

  <div class="panel-actions">
    <button onclick={onclose}>Cancel</button>
    <button
      class="primary"
      onclick={startEvolutionInitialization}
      disabled={!isFormValid || inProgress}
    >
      {#if inProgress}
        <div class="spinner"></div>
        Initializing...
      {:else}
        Start Evolution
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
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }
  label,
  .field-label {
    display: block;
    color: var(--color-overlay-text);
    font-weight: 500;
    font-size: 0.9rem;
  }
  input[type='text'],
  input[type='number'],
  select {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
    margin-top: 0.35rem;
    font-size: 0.9rem;
    box-sizing: border-box;
  }
  input[type='range'] {
    width: 100%;
    margin-top: 0.35rem;
    accent-color: var(--color-accent-primary, #6366f1);
  }
  .form-row {
    display: flex;
    gap: 1rem;
  }
  .form-row > label {
    flex: 1;
  }
  .noise-sliders {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    background: rgba(255, 255, 255, 0.02);
    padding: 0.75rem;
    border-radius: 0.375rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
  }
  .slider-label {
    display: flex;
    flex-direction: column;
  }
  .slider-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: var(--color-overlay-text-muted);
  }
  .loading-layers-box {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--color-overlay-text-muted);
    font-size: 0.85rem;
    padding: 0.5rem;
  }
  .alert-box {
    padding: 0.6rem 0.8rem;
    border-radius: 0.375rem;
    font-size: 0.8rem;
    line-height: 1.4;
  }
  .alert-box.info {
    background: rgba(99, 102, 241, 0.15);
    border: 1px dashed rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
  }
  .status-msg {
    margin-top: 0.35rem;
    padding: 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.85rem;
  }
  .status-msg.success {
    background: rgba(16, 185, 129, 0.1);
    color: #34d399;
  }
  .status-msg.error {
    background: rgba(239, 68, 68, 0.1);
    color: #f87171;
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
