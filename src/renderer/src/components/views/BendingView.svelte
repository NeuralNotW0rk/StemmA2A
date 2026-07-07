<!-- src/renderer/src/components/views/BendingView.svelte -->
<script lang="ts">
  import { onMount } from 'svelte'
  import NodeSelector from '../NodeSelector.svelte'
  import type { NodeData } from '../../utils/forms'
  import type { ErrorInfo } from '../../utils/types'

  let { modelElement, onclose, onrefresh, onError } = $props<{
    modelElement: any
    onclose: () => void
    onrefresh: () => void
    onError: (error: ErrorInfo) => void
  }>()

  let name = $state('')
  let inProgress = $state(false)
  let layers = $state<any[]>([])
  let loadingLayers = $state(true)

  // Bending Element configuration state
  let selectedAddress = $state('')
  let kernelType = $state('erode')
  
  // Dynamic Parameters
  let radius = $state(1)
  let scaleFactor = $state(1.0)
  let angle = $state(0.0)
  let offsetX = $state(0.0)
  let offsetY = $state(0.0)
  let multiplier = $state(1.0)
  
  // Target Configuration
  let targetType = $state<'indices' | 'cluster'>('indices')
  let rawIndices = $state('')
  let numClusters = $state(5)
  let selectedCluster = $state(0)

  let selectedLayer = $derived(layers.find(l => l.address === selectedAddress))

  onMount(async () => {
    if (!modelElement?.id) {
      loadingLayers = false
      return
    }
    
    try {
      const response = await window.api.getModelLayers(modelElement.id)
      if (response && response.success && Array.isArray(response.layers)) {
        layers = response.layers
        if (layers.length > 0) {
          selectedAddress = layers[0].address
        }
      }
    } catch (e) {
      console.error('Failed to load model layers:', e)
    } finally {
      loadingLayers = false
    }
  })

  // Auto-generate name based on config
  $effect(() => {
    if (selectedAddress) {
      const shortAddr = selectedAddress.split('.').pop() || selectedAddress
      name = `${modelElement.name || 'Model'} - ${shortAddr} [${kernelType}]`
    }
  })

  let isFormValid = $derived(
    name.trim() !== '' && 
    selectedAddress !== '' && 
    !loadingLayers
  )

  async function createGrating(): Promise<void> {
    if (!isFormValid || !modelElement) return

    inProgress = true
    try {
      // Gather kernel params
      const params: Record<string, any> = {}
      if (kernelType === 'erode' || kernelType === 'dilate') {
        params['radius'] = radius
      } else if (kernelType === 'scale') {
        params['scale_factor'] = scaleFactor
      } else if (kernelType === 'rotate') {
        params['angle'] = angle
      } else if (kernelType === 'translate') {
        params['offset_x'] = offsetX
        params['offset_y'] = offsetY
      } else if (kernelType === 'scalar-multiply') {
        params['multiplier'] = multiplier
      }

      // Resolve targets
      let indices: number[] = []
      if (targetType === 'indices' && rawIndices.trim() !== '') {
        indices = rawIndices
          .split(',')
          .map(s => parseInt(s.trim()))
          .filter(n => !isNaN(n))
      }

      const payload = {
        model_id: modelElement.id,
        name: name,
        elements: [
          {
            address: selectedAddress,
            kernel_type: kernelType,
            params: params,
            indices: targetType === 'indices' ? indices : [],
            perform_clustering: targetType === 'cluster',
            num_clusters: targetType === 'cluster' ? numClusters : null,
            cluster: targetType === 'cluster' ? selectedCluster : null
          }
        ]
      }

      await window.api.createGrating(payload)
      onclose()
      onrefresh()
    } catch (e) {
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
    <div class="field-group">
      <span class="field-label">Target Model</span>
      <div class="read-only-box">{modelElement?.name || 'Loading model...'}</div>
    </div>

    <label>
      Grating Name
      <input type="text" bind:value={name} placeholder="e.g., Erode Layer 4" />
    </label>

    <label>
      Target Layer
      {#if loadingLayers}
        <div class="loading-layers-box">
          <div class="spinner tiny"></div> Inspecting model structure...
        </div>
      {:else}
        <select bind:value={selectedAddress}>
          {#each layers as layer}
            <option value={layer.address}>
              {layer.address} ({layer.type}) {layer.shape ? `- Shape: [${layer.shape.join(', ')}]` : ''}
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

    <!-- Dynamic Kernel Params -->
    {#if kernelType === 'erode' || kernelType === 'dilate'}
      <label>
        Radius (Kernel footprint size)
        <input type="number" min="1" max="15" bind:value={radius} />
      </label>
    {:else if kernelType === 'scale'}
      <label>
        Scale Factor
        <input type="number" step="0.1" bind:value={scaleFactor} />
      </label>
    {:else if kernelType === 'rotate'}
      <label>
        Rotation Angle (degrees)
        <input type="number" step="1.0" bind:value={angle} />
      </label>
    {:else if kernelType === 'translate'}
      <div class="horizontal-fields">
        <label>
          Offset X
          <input type="number" step="0.5" bind:value={offsetX} />
        </label>
        <label>
          Offset Y
          <input type="number" step="0.5" bind:value={offsetY} />
        </label>
      </div>
    {:else if kernelType === 'scalar-multiply'}
      <label>
        Scale Multiplier
        <input type="number" step="0.1" bind:value={multiplier} />
      </label>
    {/if}

    <!-- Channel Target / Clustering selection -->
    <div class="target-type-selector">
      <span class="field-label">Target Options</span>
      <div class="tabs">
        <button 
          class="tab-btn" 
          class:active={targetType === 'indices'} 
          onclick={() => targetType = 'indices'}
        >
          Explicit Channels
        </button>
        <button 
          class="tab-btn" 
          class:active={targetType === 'cluster'} 
          onclick={() => targetType = 'cluster'}
        >
          Unsupervised Clustering
        </button>
      </div>
    </div>

    {#if targetType === 'indices'}
      <label>
        Target Feature Indices
        <input 
          type="text" 
          bind:value={rawIndices} 
          placeholder="e.g., 0, 1, 2, 3 (leave empty to apply to all channels)" 
        />
      </label>
    {:else}
      <div class="clustering-group">
        <div class="alert-box info">
          Dynamic Exemplar training runs a few forward passes, trains a CNN classifier on features, and runs K-Means in the backend.
        </div>
        <div class="horizontal-fields">
          <label>
            Number of Clusters
            <input type="number" min="2" max="20" bind:value={numClusters} />
          </label>
          <label>
            Target Cluster ID
            <input type="number" min="0" max={numClusters - 1} bind:value={selectedCluster} />
          </label>
        </div>
      </div>
    {/if}
  </div>

  <div class="panel-actions">
    <button onclick={onclose}>Cancel</button>
    <button class="primary" onclick={createGrating} disabled={!isFormValid || inProgress}>
      {#if inProgress}
        <div class="spinner"></div> Creating...
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
  label, .field-label {
    display: block;
    color: var(--color-overlay-text);
    font-weight: 500;
    font-size: 0.9rem;
  }
  input, select, .read-only-box {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
    margin-top: 0.35rem;
    font-size: 0.9rem;
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
  .horizontal-fields {
    display: flex;
    gap: 0.75rem;
  }
  .horizontal-fields label {
    flex: 1;
  }
  .target-type-selector {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .tabs {
    display: flex;
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.375rem;
    overflow: hidden;
  }
  .tab-btn {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--color-overlay-text-muted);
    padding: 0.5rem;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.2s ease;
  }
  .tab-btn.active {
    background: var(--color-accent-primary, #6366f1);
    color: white;
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
