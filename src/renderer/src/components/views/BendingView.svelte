<!-- src/renderer/src/components/views/BendingView.svelte -->
<script lang="ts">
  import NodeSelector from '../NodeSelector.svelte'
  import type { NodeData } from '../../utils/forms'
  import type { PanelMessage } from '../../utils/types'

  interface ModelLayer {
    address: string
    type: string
    shape?: number[]
    // Whether the layer outputs 2D feature maps (B, C, H, W), which spatial kernels require
    spatial_output: boolean
  }

  interface KernelOption {
    value: string
    label: string
    // Spatial kernels raise an error on anything but 2D feature maps
    spatial: boolean
    // Shown in the tip view while the option is hovered, and while it is selected
    description: string
  }

  const SPATIAL_NOTE = ' Needs a layer with 2D spatial output.'
  const KERNELS: KernelOption[] = [
    {
      value: 'erode',
      label: 'Erode (Morphological Erosion)',
      spatial: true,
      description:
        'Shrinks the strong regions of each feature map, thinning out whatever features they draw.' +
        SPATIAL_NOTE
    },
    {
      value: 'dilate',
      label: 'Dilate (Morphological Dilation)',
      spatial: true,
      description:
        'Grows the strong regions of each feature map, thickening whatever features they draw.' +
        SPATIAL_NOTE
    },
    {
      value: 'scale',
      label: 'Scale (2D Spatial Zoom)',
      spatial: true,
      description: 'Zooms each feature map in or out around its center.' + SPATIAL_NOTE
    },
    {
      value: 'rotate',
      label: 'Rotate (2D Spatial Rotation)',
      spatial: true,
      description: 'Rotates each feature map around its center.' + SPATIAL_NOTE
    },
    {
      value: 'translate',
      label: 'Translate (2D Spatial Translation)',
      spatial: true,
      description: 'Shifts each feature map horizontally and vertically.' + SPATIAL_NOTE
    },
    {
      value: 'scalar-multiply',
      label: 'Scalar Multiply (Gain/Ablation)',
      spatial: false,
      description:
        'Multiplies the chosen channels by a factor: above 1 boosts them, between 0 and 1 weakens them and 0 silences them. Works on any layer.'
    },
    {
      value: 'ablate',
      label: 'Ablate (Zero Ablation)',
      spatial: false,
      description: 'Sets the chosen channels to zero, removing their contribution. Works on any layer.'
    }
  ]

  let { modelElement, onclose, onrefresh, onError }: {
    modelElement: NodeData | null
    onclose: () => void
    onrefresh: () => void
    onError: (error: PanelMessage) => void
  } = $props()

  let selectedModel = $state<NodeData | null>(null)

  $effect((): void => {
    selectedModel = modelElement
  })
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
  let selectedKernel = $derived(KERNELS.find((k): boolean => k.value === kernelType))
  // The field's own tip, then the selected kernel's on its own line, as in DynamicForm selects
  let kernelTip = $derived(
    selectedKernel
      ? `${TIPS.kernel}
${selectedKernel.label.split(' (')[0]}: ${selectedKernel.description}`
      : TIPS.kernel
  )
  // The selection is kept, not switched, so the user sees why Create is disabled
  let kernelUnsupported = $derived(
    !!selectedLayer && !!selectedKernel?.spatial && !selectedLayer.spatial_output
  )

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

  let isFormValid = $derived(
    selectedAddress !== '' && !loadingLayers && !!selectedModel && !kernelUnsupported
  )

  import { addJob, pollJobStatus } from '../../utils/job-management'

  // Shown in the tip view while each field is hovered or focused
  const TIPS = {
    model:
      'The model this grating bends. Bending changes what a layer outputs while the model generates; the model itself is never altered. The new grating is bound to this model and only works with it.',
    name: "Name shown on the grating's node. Left empty, the grating is named after its layer and kernel, for example 'conv1.conv [erode]'.",
    layer:
      'The layer whose output the kernel changes. In image models, earlier layers tend to shape broad structure and later layers finer detail. Erode, Dilate, Scale, Rotate and Translate need a layer with 2D spatial output, such as a convolution in an image model.',
    kernel:
      "How the layer's output is changed. Kernels the selected layer can't take are disabled. Amounts are set in Generate.",
    clustering:
      "Groups the layer's channels by how they respond, so the grating can bend one group instead of the whole layer. It runs a few test generations, learns a profile of each channel and sorts the channels with k-means, which makes creating the grating slower. Pick a cluster in Generate.",
    clusters:
      "How many groups to sort the layer's channels into. More clusters give smaller, more specific groups; fewer give broader ones. Each cluster can be targeted on its own in Generate. Greyed out until Precompute Feature Clustering is on."
  } as const

  async function createGrating(): Promise<void> {
    if (!isFormValid || !selectedModel) return

    inProgress = true
    try {
      let finalName = name.trim()
      if (!finalName) {
        finalName = `${selectedAddress} [${kernelType}]`
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

      const res = (await window.api.createGrating(payload)) as { success?: boolean; job_id?: string }
      if (res?.job_id) {
        addJob(`Create Grating (${finalName})`, { model_id: selectedModel.id }, 'running', res.job_id)
        onclose()
        pollJobStatus(res.job_id)
          .then((): void => {
            onrefresh()
          })
          .catch((err: unknown): void => {
            console.error('Background grating creation failed:', err)
          })
        return
      }

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
    <div data-tip={TIPS.model} data-tip-title="Target Model">
      <NodeSelector
        label="Target Model"
        filter={{ type: 'model' }}
        bind:node={selectedModel}
        id="target-model-selector"
      />
    </div>

    <label data-tip={TIPS.name} data-tip-title="Grating Name">
      Grating Name
      <input type="text" bind:value={name} placeholder="e.g., Erode Layer 4" />
    </label>

    <label data-tip={TIPS.layer} data-tip-title="Target Layer">
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

    <label data-tip={kernelTip} data-tip-title="Bending Kernel">
      Bending Kernel
      <select bind:value={kernelType}>
        {#each KERNELS as kernel (kernel.value)}
          <option
            value={kernel.value}
            disabled={kernel.spatial && !!selectedLayer && !selectedLayer.spatial_output}
            data-tip={kernel.description}
            data-tip-title={kernel.label.split(' (')[0]}
          >
            {kernel.label}
          </option>
        {/each}
      </select>
      {#if kernelUnsupported && selectedKernel}
        <span class="field-error">
          {selectedKernel.label.split(' (')[0]} needs a layer with 2D spatial output. Choose Scalar Multiply
          or Ablate, or a convolution layer of an image model.
        </span>
      {/if}
    </label>

    <!-- Feature Clustering Pre-computation Toggle -->
    <div
      style="margin-top: 0.5rem;"
      data-tip={TIPS.clustering}
      data-tip-title="Precompute Feature Clustering"
    >
      <label
        style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; color: var(--color-overlay-text);"
      >
        <input
          type="checkbox"
          bind:checked={performClustering}
          style="width: auto; margin-top: 0;"
        />
        Precompute Feature Clustering
      </label>
    </div>

    <!-- Greyed out rather than hidden: the checkbox is a setting, not an attached artifact -->
    <div class="clustering-group" class:disabled={!performClustering}>
      <label data-tip={TIPS.clusters} data-tip-title="Number of Clusters">
        Number of Clusters
        <input
          type="number"
          min="2"
          max="20"
          bind:value={numClusters}
          disabled={!performClustering}
        />
      </label>
    </div>
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
  label {
    display: block;
    color: var(--color-overlay-text);
    font-weight: 500;
    font-size: 0.9rem;
  }
  .field-error {
    display: block;
    margin-top: 0.375rem;
    color: var(--color-error);
    font-size: 0.8rem;
    font-weight: 400;
    line-height: 1.4;
  }
  input,
  select {
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
  .clustering-group.disabled {
    opacity: 0.45;
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
