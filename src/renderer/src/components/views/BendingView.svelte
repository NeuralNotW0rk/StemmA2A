<!-- src/renderer/src/components/views/BendingView.svelte -->
<script lang="ts">
  import NodeSelector from '../NodeSelector.svelte'
  import type { NodeData } from '../../utils/forms'
  import type { ErrorInfo } from '../../utils/types'

  interface ModelLayer {
    address: string
    type: string
    shape?: number[]
  }

  let { modelElement, onclose, onrefresh, onError } = $props<{
    modelElement: NodeData | null
    onclose: () => void
    onrefresh: () => void
    onError: (error: ErrorInfo) => void
  }>()

  let selectedModel = $state<NodeData | null>(modelElement)
  let name = $state('')
  let inProgress = $state(false)
  let layers = $state<ModelLayer[]>([])
  let loadingLayers = $state(true)

  // Bending Element configuration state
  let selectedAddress = $state('')
  let kernelType = $state('erode')

  // Clustering Configuration
  let performClustering = $state(false)
  let numClusters = $state(5)

  let selectedLayer = $derived(layers.find((l): boolean => l.address === selectedAddress))

  $effect((): (() => void) | void => {
    const modelId = selectedModel?.id
    if (!modelId) {
      layers = []
      selectedAddress = ''
      loadingLayers = false
      return
    }

    let isCurrent = true
    loadingLayers = true

    window.api.getModelLayers(modelId)
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
          const fetchedLayers = (response as Record<string, unknown>).layers as ModelLayer[]
          layers = fetchedLayers
          if (layers.length > 0) {
            selectedAddress = layers[0].address
          } else {
            selectedAddress = ''
          }
        } else {
          layers = []
          selectedAddress = ''
        }
      })
      .catch((e: unknown): void => {
        if (!isCurrent) return
        const message = e instanceof Error ? e.message : String(e)
        console.error('Failed to load model layers:', message)
        layers = []
        selectedAddress = ''
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

  let isFormValid = $derived(selectedAddress !== '' && !loadingLayers && !!selectedModel)

  async function createGrating(): Promise<void> {
    if (!isFormValid || !selectedModel) return

    inProgress = true
    try {
      let finalName = name.trim()
      if (!finalName) {
        finalName = `${selectedModel.name || 'Model'} - ${selectedAddress} [${kernelType}]`
      }

      const payload = {
        model_id: selectedModel.id,
        name: finalName,
        elements: [
          {
            address: selectedAddress,
            kernel_type: kernelType,
            params: {},
            indices: [],
            perform_clustering: performClustering,
            num_clusters: performClustering ? numClusters : null,
            cluster: null
          }
        ]
      }

      await window.api.createGrating(payload)
      onclose()
      onrefresh()
    } catch (e: unknown) {
      console.error('Failed to create grating:', e)
      const message = e instanceof Error ? e.message : String(e)
      onError({ title: 'Grating Creation Failed', message })
    } finally {
      inProgress = false
    }
  }
</script>

<div class="view-container">
  <div class="view-content">
    <NodeSelector
      label="Target Model"
      filter={{ type: 'model' }}
      bind:node={selectedModel}
      id="target-model-selector"
    />

    <label>
      Grating Name
      <input type="text" bind:value={name} placeholder="e.g., Erode Layer 4" />
    </label>

    <label>
      Target Layer
      {#if loadingLayers}
        <div class="loading-layers-box">
          <div class="spinner tiny"></div>
          Inspecting model structure...
        </div>
      {:else}
        <select bind:value={selectedAddress}>
          {#each layers as layer (layer.address)}
            <option value={layer.address}>
              {layer.address} ({layer.type}) {layer.shape
                ? `- Shape: [${layer.shape.join(', ')}]`
                : ''}
            </option>
          {/each}
        </select>
      {/if}
    </label>

    <label>
      Bending Kernel
      <select bind:value={kernelType}>
        <option value="erode">Erode (Morphological Erosion)</option>
        <option value="dilate">Dilate (Morphological Dilation)</option>
        <option value="scale">Scale (2D Spatial Zoom)</option>
        <option value="rotate">Rotate (2D Spatial Rotation)</option>
        <option value="translate">Translate (2D Spatial Translation)</option>
        <option value="scalar-multiply">Scalar Multiply (Gain/Ablation)</option>
        <option value="ablate">Ablate (Zero Ablation)</option>
      </select>
    </label>

    <!-- Feature Clustering Pre-computation Toggle -->
    <div style="margin-top: 0.5rem;">
      <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; color: var(--color-overlay-text);">
        <input type="checkbox" bind:checked={performClustering} style="width: auto; margin-top: 0;" />
        Precompute Feature Clustering
      </label>
    </div>

    {#if performClustering}
      <div class="clustering-group">
        <div class="alert-box info">
          Dynamic Exemplar training runs a few forward passes, trains a CNN classifier on features,
          and runs K-Means in the backend to partition channel activities.
        </div>
        <label>
          Number of Clusters
          <input type="number" min="2" max="20" bind:value={numClusters} />
        </label>
      </div>
    {/if}
  </div>

  <div class="panel-actions">
    <button onclick={onclose}>Cancel</button>
    <button class="primary" onclick={createGrating} disabled={!isFormValid || inProgress}>
      {#if inProgress}
        <div class="spinner"></div>
        Creating...
      {:else}
        Create Grating
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
  input,
  select,
  .read-only-box {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
    margin-top: 0.35rem;
    font-size: 0.9rem;
  }
  select option {
    background: var(--color-background-soft, #222222);
    color: var(--color-overlay-text, #ffffff);
  }
  .read-only-box {
    background: rgba(255, 255, 255, 0.05);
    border-color: transparent;
  }
  .loading-layers-box {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--color-overlay-text-muted);
    font-size: 0.85rem;
    padding: 0.5rem;
  }
  .clustering-group {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .alert-box {
    padding: 0.6rem 0.8rem;
    border-radius: 0.375rem;
    font-size: 0.8rem;
    line-height: 1.3;
  }
  .alert-box.info {
    background: rgba(99, 102, 241, 0.15);
    border: 1px dashed rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
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
