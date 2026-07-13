<!-- src/renderer/src/components/views/OperationView.svelte -->
<script lang="ts">
  import { tick, onDestroy, untrack, type Snippet } from 'svelte'
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

  let initiatorType = $derived.by(() => {
    let type = $initiatorNodeStore?.type
    if ($contextStore) {
      if ($contextStore.init_audio || $contextStore.source_audio || $contextStore.source_audio_id) {
        type = 'audio'
      } else if ($contextStore.init_latent) {
        type = 'latent'
      } else if ($contextStore.gratings && $contextStore.gratings.length > 0) {
        type = 'grating'
      } else {
        type = 'model'
      }
    }
    return type
  })

  let outputType = $derived.by(() => {
    const node = $initiatorNodeStore
    if (!node) return null
    if (node.type === 'batch') {
      const memberIds = node.member_ids || []
      if (memberIds.length > 0 && $cyInstanceStore) {
        const firstMember = $cyInstanceStore.getElementById(memberIds[0])
        if (firstMember && firstMember.length > 0) {
          return firstMember.data('output_type') || null
        }
      }
    }
    return node.output_type || null
  })

  let activeDescription = $derived.by(() => {
    if (!$selectedOperation) return ''
    if (initiatorType === 'model') {
      if (outputType) {
        return `Generate ${outputType} from the selected model`
      } else {
        return $selectedOperation.description || `Generate output from the selected model`
      }
    }
    if (
      initiatorType &&
      $selectedOperation.context_overrides &&
      $selectedOperation.context_overrides[initiatorType]
    ) {
      const override = $selectedOperation.context_overrides[initiatorType]
      return override.description || $selectedOperation.description
    }
    return $selectedOperation.description
  })

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

  // Automatically initialize overrides on selected gratings when nodes are chosen
  $effect(() => {
    for (const item of selectedGratings) {
      if (item.node && typeof item.node === 'object') {
        const nodeObj = item.node as any
        const nodeId = nodeObj.id
        if (item.loadedNodeId !== nodeId) {
          item.loadedNodeId = nodeId
          const elements = (nodeObj.elements || []) as any[]
          item.overrides = elements.map((el) => {
            const ktype = el.kernel_type
            const meta = el.metadata || {}

            let targetType: 'all' | 'indices' | 'cluster' = 'all'
            if (meta.indices && Array.isArray(meta.indices) && meta.indices.length > 0) {
              targetType = 'indices'
            } else if (meta.cluster !== null && meta.cluster !== undefined) {
              targetType = 'cluster'
            } else if (meta.cluster_map) {
              targetType = 'cluster'
            }

            const params: Record<string, any> = {}
            if (ktype === 'erode' || ktype === 'dilate') {
              params.radius = meta.radius ?? 1
            } else if (ktype === 'scale') {
              params.scale_factor = meta.scale_factor ?? 1.0
            } else if (ktype === 'rotate') {
              params.angle = meta.angle ?? 0.0
            } else if (ktype === 'translate') {
              params.offset_x = meta.offset_x ?? 0.0
              params.offset_y = meta.offset_y ?? 0.0
            } else if (ktype === 'scalar-multiply') {
              params.factor = meta.factor ?? meta.multiplier ?? 1.0
            } else if (ktype === 'resize') {
              params.scale_x = meta.scale_x ?? 1.0
              params.scale_y = meta.scale_y ?? 1.0
            } else if (ktype === 'binary-thresh') {
              params.threshold = meta.threshold ?? 0.0
            }

            return {
              address: el.address,
              kernel_type: ktype,
              targetType,
              indicesText: meta.indices ? meta.indices.join(', ') : '',
              cluster: meta.cluster ?? 0,
              params,
              batchFields: {}
            }
          })
        }
      } else {
        item.loadedNodeId = undefined
        item.overrides = undefined
      }
    }
  })

  function toggleGratingBatchMode(ov: any, fieldName: string): void {
    if (!ov.batchFields) {
      ov.batchFields = {}
    }
    ov.batchFields[fieldName] = !ov.batchFields[fieldName]
  }

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
          const contextGratings = $contextStore.gratings as Array<{ id: string; strength: number; overrides?: any[] }>
          selectedGratings = contextGratings.map((g, index) => {
            const node = $cyInstanceStore.$id(g.id).data()
            const item: NodeListItem = {
              id: index,
              node: node || g.id,
              strength: g.strength
            }
            if (g.overrides) {
              item.overrides = g.overrides.map((ov) => {
                const meta = ov.metadata || {}
                const el = node?.elements?.find((e: any) => e.address === ov.address)
                const ktype = el?.kernel_type || 'erode'
                
                let targetType: 'all' | 'indices' | 'cluster' = 'all'
                if (meta.indices && Array.isArray(meta.indices) && meta.indices.length > 0) {
                  targetType = 'indices'
                } else if (meta.cluster !== null && meta.cluster !== undefined) {
                  targetType = 'cluster'
                }
                
                const params: Record<string, any> = {}
                if (ktype === 'erode' || ktype === 'dilate') {
                  params.radius = meta.radius ?? el?.metadata?.radius ?? 1
                } else if (ktype === 'scale') {
                  params.scale_factor = meta.scale_factor ?? el?.metadata?.scale_factor ?? 1.0
                } else if (ktype === 'rotate') {
                  params.angle = meta.angle ?? el?.metadata?.angle ?? 0.0
                } else if (ktype === 'translate') {
                  params.offset_x = meta.offset_x ?? el?.metadata?.offset_x ?? 0.0
                  params.offset_y = meta.offset_y ?? el?.metadata?.offset_y ?? 0.0
                } else if (ktype === 'scalar-multiply') {
                  params.factor = meta.factor ?? meta.multiplier ?? el?.metadata?.factor ?? el?.metadata?.multiplier ?? 1.0
                } else if (ktype === 'resize') {
                  params.scale_x = meta.scale_x ?? el?.metadata?.scale_x ?? 1.0
                  params.scale_y = meta.scale_y ?? el?.metadata?.scale_y ?? 1.0
                } else if (ktype === 'binary-thresh') {
                  params.threshold = meta.threshold ?? el?.metadata?.threshold ?? 0.0
                }

                return {
                  address: ov.address,
                  kernel_type: ktype,
                  targetType,
                  indicesText: meta.indices ? meta.indices.join(', ') : '',
                  cluster: meta.cluster ?? 0,
                  params,
                  batchFields: ov.batchFields || {}
                }
              })
              item.loadedNodeId = node?.id || g.id
            }
            return item
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
          isReplicated ? null : $initiatorNodeStore,
          formData
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
      const gratings = validGratings.map((l) => {
        const id = typeof l.node === 'string' ? l.node : l.node?.id
        const strength = l.strength ?? 1.0
        
        const overrides: any[] = []
        if (l.overrides) {
          for (const o of l.overrides) {
            let indices: number[] = []
            if (o.targetType === 'indices' && o.indicesText.trim() !== '') {
              indices = o.indicesText
                .split(',')
                .map((s) => parseInt(s.trim()))
                .filter((n) => !isNaN(n))
            }
            
            const metaOverride: Record<string, any> = {
              indices: o.targetType === 'indices' ? indices : [],
              cluster: o.targetType === 'cluster' ? o.cluster : null,
              ...o.params
            }
            
            overrides.push({
              address: o.address,
              metadata: metaOverride
            })
          }
        }

        return {
          id,
          strength,
          overrides
        }
      })
      if (gratings.length > 0) {
        basePayload.gratings = gratings
      }
    }

    // Collect batch fields from grating overrides
    const gratingBatchParams: Record<string, (string | number)[]> = {}
    selectedGratings.forEach((l, gIdx) => {
      if (l.overrides) {
        l.overrides.forEach((o, oIdx) => {
          if (o.batchFields) {
            for (const key in o.batchFields) {
              if (o.batchFields[key]) {
                const tempKey = `grating__${gIdx}__${oIdx}__${key}`
                let rawVal = ''
                if (key === 'cluster') {
                  rawVal = String(o.cluster)
                } else if (key === 'indicesText') {
                  rawVal = String(o.indicesText)
                } else {
                  rawVal = String(o.params[key])
                }
                gratingBatchParams[tempKey] = parseSequence(rawVal)
              }
            }
          }
        })
      }
    })

    const hasBatch = batchFields.size > 0 || Object.keys(gratingBatchParams).length > 0

    if (!hasBatch) {
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
      const batchParams: Record<string, (string | number)[]> = { ...gratingBatchParams }
      const staticParams: Record<string, unknown> = {}

      for (const key in basePayload) {
        if (batchFields.has(key)) {
          batchParams[key] = parseSequence(String(basePayload[key]))
        } else {
          staticParams[key] = basePayload[key]
        }
      }

      // Construct a clean representation of gratings with non-batched parameters
      const staticGratings = selectedGratings.filter((l) => l.node).map((l) => {
        const id = typeof l.node === 'string' ? l.node : l.node?.id
        const strength = l.strength ?? 1.0
        
        const overrides: any[] = []
        if (l.overrides) {
          for (const o of l.overrides) {
            let indices: number[] = []
            if (o.targetType === 'indices' && !o.batchFields?.indicesText && o.indicesText.trim() !== '') {
              indices = o.indicesText
                .split(',')
                .map((s) => parseInt(s.trim()))
                .filter((n) => !isNaN(n))
            }
            
            const metaOverride: Record<string, any> = {
              indices: o.targetType === 'indices' && !o.batchFields?.indicesText ? indices : [],
              cluster: o.targetType === 'cluster' && !o.batchFields?.cluster ? o.cluster : null,
            }
            
            for (const paramKey in o.params) {
              if (!o.batchFields?.[paramKey]) {
                metaOverride[paramKey] = o.params[paramKey]
              }
            }
            
            overrides.push({
              address: o.address,
              metadata: metaOverride
            })
          }
        }
        
        return {
          id,
          strength,
          overrides
        }
      })

      const paramNames = Object.keys(batchParams)
      const paramValues = Object.values(batchParams)
      const combinations = cartesian(...paramValues)

      console.log(`Starting batch execution with ${combinations.length} combinations.`)

      const batchId =
        parentBatchId && isReplicated && addToSameBatch
          ? parentBatchId
          : `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`

      combinations.forEach((combination) => {
        // Deep clone static gratings so each combination gets its own overrides
        const combinationGratings = JSON.parse(JSON.stringify(staticGratings))
        const batchPayload: Record<string, unknown> = { ...staticParams }
        
        combination.forEach((value, index) => {
          const paramName = paramNames[index]
          if (paramName.startsWith('grating__')) {
            const [, gIdxStr, oIdxStr, key] = paramName.split('__')
            const gIdx = parseInt(gIdxStr)
            const oIdx = parseInt(oIdxStr)
            
            const override = combinationGratings[gIdx]?.overrides?.[oIdx]
            if (override) {
              if (key === 'cluster') {
                override.metadata.cluster = value
              } else if (key === 'indicesText') {
                let indices: number[] = []
                if (typeof value === 'string') {
                  indices = value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
                } else if (typeof value === 'number') {
                  indices = [value]
                }
                override.metadata.indices = indices
              } else {
                override.metadata[key] = value
              }
            }
          } else {
            batchPayload[paramName] = value
          }
        })
        
        batchPayload.gratings = combinationGratings

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

{#snippet gratingExtra(item: any, index: number)}
  {#if item.overrides && item.overrides.length > 0}
    <div class="grating-overrides-form">
      {#each item.overrides as ov, ovIdx (ov.address)}
        <div class="grating-element-override">
          <div class="element-header">
            <span class="element-address">{ov.address}</span>
            <span class="element-kernel-badge">{ov.kernel_type}</span>
          </div>
          
          <div class="element-params">
            <!-- Dynamic parameters depending on kernel type -->
            {#if ov.kernel_type === 'erode' || ov.kernel_type === 'dilate'}
              <div class="sub-label-row">
                <span class="sub-label">Radius</span>
                {#if !ov.batchFields?.radius}
                  <span class="sub-value">{ov.params.radius ?? 1}</span>
                {/if}
              </div>
              <div class="sub-input-row">
                {#if ov.batchFields?.radius}
                  <input
                    type="text"
                    bind:value={ov.params.radius}
                    placeholder="e.g. 1, 2, 3-5"
                  />
                {:else}
                  <input
                    type="range"
                    min="1"
                    max="15"
                    step="1"
                    bind:value={ov.params.radius}
                  />
                {/if}
                <div class="sub-field-actions">
                  <button
                    type="button"
                    class="sub-action-btn"
                    onclick={() => toggleGratingBatchMode(ov, 'radius')}
                    title="Toggle sequence mode"
                  >
                    {ov.batchFields?.radius ? '−' : '+'}
                  </button>
                </div>
              </div>
            {:else if ov.kernel_type === 'scale'}
              <span class="sub-label">Scale Factor</span>
              <div class="sub-input-row">
                {#if ov.batchFields?.scale_factor}
                  <input
                    type="text"
                    bind:value={ov.params.scale_factor}
                    placeholder="e.g. 0.5, 1.0, 1.2-2.0:0.2"
                  />
                {:else}
                  <input
                    type="number"
                    step="0.1"
                    bind:value={ov.params.scale_factor}
                  />
                {/if}
                <div class="sub-field-actions">
                  <button
                    type="button"
                    class="sub-action-btn"
                    onclick={() => toggleGratingBatchMode(ov, 'scale_factor')}
                    title="Toggle sequence mode"
                  >
                    {ov.batchFields?.scale_factor ? '−' : '+'}
                  </button>
                </div>
              </div>
            {:else if ov.kernel_type === 'rotate'}
              <span class="sub-label">Rotation Angle (degrees)</span>
              <div class="sub-input-row">
                {#if ov.batchFields?.angle}
                  <input
                    type="text"
                    bind:value={ov.params.angle}
                    placeholder="e.g. 0, 90, 180-360:90"
                  />
                {:else}
                  <input
                    type="number"
                    step="1.0"
                    bind:value={ov.params.angle}
                  />
                {/if}
                <div class="sub-field-actions">
                  <button
                    type="button"
                    class="sub-action-btn"
                    onclick={() => toggleGratingBatchMode(ov, 'angle')}
                    title="Toggle sequence mode"
                  >
                    {ov.batchFields?.angle ? '−' : '+'}
                  </button>
                </div>
              </div>
            {:else if ov.kernel_type === 'translate'}
              <div class="horizontal-fields">
                <div style="flex: 1;">
                  <span class="sub-label">Offset X</span>
                  <div class="sub-input-row">
                    {#if ov.batchFields?.offset_x}
                      <input
                        type="text"
                        bind:value={ov.params.offset_x}
                        placeholder="e.g. 0, 1, 2"
                      />
                    {:else}
                      <input
                        type="number"
                        step="0.5"
                        bind:value={ov.params.offset_x}
                      />
                    {/if}
                    <div class="sub-field-actions">
                      <button
                        type="button"
                        class="sub-action-btn"
                        onclick={() => toggleGratingBatchMode(ov, 'offset_x')}
                        title="Toggle sequence mode"
                      >
                        {ov.batchFields?.offset_x ? '−' : '+'}
                      </button>
                    </div>
                  </div>
                </div>
                <div style="flex: 1;">
                  <span class="sub-label">Offset Y</span>
                  <div class="sub-input-row">
                    {#if ov.batchFields?.offset_y}
                      <input
                        type="text"
                        bind:value={ov.params.offset_y}
                        placeholder="e.g. 0, 1, 2"
                      />
                    {:else}
                      <input
                        type="number"
                        step="0.5"
                        bind:value={ov.params.offset_y}
                      />
                    {/if}
                    <div class="sub-field-actions">
                      <button
                        type="button"
                        class="sub-action-btn"
                        onclick={() => toggleGratingBatchMode(ov, 'offset_y')}
                        title="Toggle sequence mode"
                      >
                        {ov.batchFields?.offset_y ? '−' : '+'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            {:else if ov.kernel_type === 'scalar-multiply'}
              <span class="sub-label">Scale Multiplier (Factor)</span>
              <div class="sub-input-row">
                {#if ov.batchFields?.factor}
                  <input
                    type="text"
                    bind:value={ov.params.factor}
                    placeholder="e.g. 0.5, 1.0, 1.5-3.0:0.5"
                  />
                {:else}
                  <input
                    type="number"
                    step="0.1"
                    bind:value={ov.params.factor}
                  />
                {/if}
                <div class="sub-field-actions">
                  <button
                    type="button"
                    class="sub-action-btn"
                    onclick={() => toggleGratingBatchMode(ov, 'factor')}
                    title="Toggle sequence mode"
                  >
                    {ov.batchFields?.factor ? '−' : '+'}
                  </button>
                </div>
              </div>
            {:else if ov.kernel_type === 'resize'}
              <div class="horizontal-fields">
                <div style="flex: 1;">
                  <span class="sub-label">Scale X</span>
                  <div class="sub-input-row">
                    {#if ov.batchFields?.scale_x}
                      <input
                        type="text"
                        bind:value={ov.params.scale_x}
                        placeholder="e.g. 0.5, 1.0"
                      />
                    {:else}
                      <input
                        type="number"
                        step="0.1"
                        bind:value={ov.params.scale_x}
                      />
                    {/if}
                    <div class="sub-field-actions">
                      <button
                        type="button"
                        class="sub-action-btn"
                        onclick={() => toggleGratingBatchMode(ov, 'scale_x')}
                        title="Toggle sequence mode"
                      >
                        {ov.batchFields?.scale_x ? '−' : '+'}
                      </button>
                    </div>
                  </div>
                </div>
                <div style="flex: 1;">
                  <span class="sub-label">Scale Y</span>
                  <div class="sub-input-row">
                    {#if ov.batchFields?.scale_y}
                      <input
                        type="text"
                        bind:value={ov.params.scale_y}
                        placeholder="e.g. 0.5, 1.0"
                      />
                    {:else}
                      <input
                        type="number"
                        step="0.1"
                        bind:value={ov.params.scale_y}
                      />
                    {/if}
                    <div class="sub-field-actions">
                      <button
                        type="button"
                        class="sub-action-btn"
                        onclick={() => toggleGratingBatchMode(ov, 'scale_y')}
                        title="Toggle sequence mode"
                      >
                        {ov.batchFields?.scale_y ? '−' : '+'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            {:else if ov.kernel_type === 'binary-thresh'}
              <span class="sub-label">Threshold</span>
              <div class="sub-input-row">
                {#if ov.batchFields?.threshold}
                  <input
                    type="text"
                    bind:value={ov.params.threshold}
                    placeholder="e.g. 0.1, 0.5, 1.0"
                  />
                {:else}
                  <input
                    type="number"
                    step="0.1"
                    bind:value={ov.params.threshold}
                  />
                {/if}
                <div class="sub-field-actions">
                  <button
                    type="button"
                    class="sub-action-btn"
                    onclick={() => toggleGratingBatchMode(ov, 'threshold')}
                    title="Toggle sequence mode"
                  >
                    {ov.batchFields?.threshold ? '−' : '+'}
                  </button>
                </div>
              </div>
            {/if}
            
            <!-- Target feature / cluster selection -->
            <div class="sub-target-selector">
              <span class="sub-field-label">Target Options</span>
              <div class="sub-tabs">
                <button
                  type="button"
                  class="sub-tab-btn"
                  class:active={ov.targetType === 'all'}
                  onclick={() => (ov.targetType = 'all')}
                >
                  All Channels
                </button>
                <button
                  type="button"
                  class="sub-tab-btn"
                  class:active={ov.targetType === 'indices'}
                  onclick={() => (ov.targetType = 'indices')}
                >
                  Explicit Channels
                </button>
                {#if item.node && typeof item.node === 'object' && item.node.elements?.[ovIdx]?.metadata?.cluster_map}
                  <button
                    type="button"
                    class="sub-tab-btn"
                    class:active={ov.targetType === 'cluster'}
                    onclick={() => (ov.targetType = 'cluster')}
                  >
                    Unsupervised Clustering
                  </button>
                {/if}
              </div>
            </div>
            
            {#if ov.targetType === 'indices'}
              <span class="sub-label">Target Feature Indices</span>
              <div class="sub-input-row">
                {#if ov.batchFields?.indicesText}
                  <input
                    type="text"
                    bind:value={ov.indicesText}
                    placeholder="e.g. [0,1], [2,3]"
                  />
                {:else}
                  <input
                    type="text"
                    bind:value={ov.indicesText}
                    placeholder="e.g. 0, 1, 2, 3"
                  />
                {/if}
                <div class="sub-field-actions">
                  <button
                    type="button"
                    class="sub-action-btn"
                    onclick={() => toggleGratingBatchMode(ov, 'indicesText')}
                    title="Toggle sequence mode"
                  >
                    {ov.batchFields?.indicesText ? '−' : '+'}
                  </button>
                </div>
              </div>
            {:else if ov.targetType === 'cluster'}
              {@const clusterMap = item.node.elements?.[ovIdx]?.metadata?.cluster_map}
              {@const maxClusterId = Math.max(0, ...(clusterMap || []).map(item => Number(item?.cluster_index ?? 0)))}
              <span class="sub-label">Target Cluster ID (0 to {maxClusterId})</span>
              <div class="sub-input-row">
                {#if ov.batchFields?.cluster}
                  <input
                    type="text"
                    bind:value={ov.cluster}
                    placeholder="e.g. 0, 1, 2, 0-4"
                  />
                {:else}
                  <input
                    type="number"
                    min="0"
                    max={maxClusterId}
                    bind:value={ov.cluster}
                  />
                {/if}
                <div class="sub-field-actions">
                  <button
                    type="button"
                    class="sub-action-btn"
                    onclick={() => toggleGratingBatchMode(ov, 'cluster')}
                    title="Toggle sequence mode"
                  >
                    {ov.batchFields?.cluster ? '−' : '+'}
                  </button>
                </div>
              </div>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
{/snippet}

<div class="view-container">
  <div class="view-content">
    {#if isLoading}
      <p class="centered-text">Loading configuration...</p>
    {:else if error}
      <div class="error-message">
        <p>Error: {error}</p>
      </div>
    {:else if $selectedOperation}
      <p class="op-description">{activeDescription || 'No description available.'}</p>

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
            itemExtra={gratingExtra}
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

  /* Grating nested form styles */
  .grating-overrides-form {
    margin-top: 0.5rem;
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--color-border-glass-1);
    border-radius: 0.375rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .grating-element-override {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .element-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px dashed rgba(255, 255, 255, 0.1);
    padding-bottom: 0.25rem;
  }
  .element-address {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--color-overlay-text);
  }
  .element-kernel-badge {
    font-size: 0.7rem;
    background: var(--color-accent-primary, #6366f1);
    color: white;
    padding: 0.1rem 0.4rem;
    border-radius: 0.25rem;
    text-transform: uppercase;
    font-weight: bold;
  }
  .element-params {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .sub-label {
    display: block;
    color: var(--color-text-muted, #aaa);
    font-size: 0.8rem;
    font-weight: 500;
  }
  .sub-label input[type='number'],
  .sub-label input[type='text'] {
    width: 100%;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--color-overlay-border-primary, rgba(255, 255, 255, 0.1));
    color: var(--color-overlay-text);
    padding: 0.35rem;
    border-radius: 0.25rem;
    margin-top: 0.25rem;
    font-size: 0.8rem;
  }
  .sub-label input[type='range'] {
    width: 100%;
    margin-top: 0.25rem;
  }
  .sub-target-selector {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    margin-top: 0.25rem;
  }
  .sub-field-label {
    font-size: 0.75rem;
    color: var(--color-text-muted, #aaa);
    font-weight: 500;
  }
  .sub-tabs {
    display: flex;
    border: 1px solid var(--color-overlay-border-primary, rgba(255, 255, 255, 0.1));
    border-radius: 0.25rem;
    overflow: hidden;
  }
  .sub-tab-btn {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--color-text-muted, #aaa);
    padding: 0.3rem;
    cursor: pointer;
    font-size: 0.75rem;
    font-weight: 500;
    transition: all 0.2s ease;
  }
  .sub-tab-btn.active {
    background: var(--color-accent-primary, #6366f1);
    color: white;
  }
  .sub-tab-btn:hover:not(.active) {
    background: rgba(255, 255, 255, 0.05);
  }
  .horizontal-fields {
    display: flex;
    gap: 0.5rem;
  }
  .horizontal-fields .sub-label {
    flex: 1;
  }
  .sub-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .sub-value {
    font-size: 0.8rem;
    color: var(--color-accent-primary, #6366f1);
    font-weight: 600;
  }
  .sub-input-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .sub-input-row > input[type='text'],
  .sub-input-row > input[type='number'] {
    flex-grow: 1;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--color-overlay-border-primary, rgba(255, 255, 255, 0.1));
    color: var(--color-overlay-text);
    padding: 0.35rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
    width: 100%;
    box-sizing: border-box;
  }
  .sub-input-row > input[type='range'] {
    flex-grow: 1;
  }
  .sub-field-actions {
    display: flex;
    align-items: center;
    flex-shrink: 0;
  }
  .sub-action-btn {
    background: none;
    border: 1px solid var(--color-overlay-border-primary, rgba(255, 255, 255, 0.2));
    color: var(--color-overlay-text);
    cursor: pointer;
    width: 20px;
    height: 20px;
    min-width: 20px;
    min-height: 20px;
    border-radius: 50%;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    box-sizing: border-box;
    transition: all 0.2s ease;
  }
  .sub-action-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: var(--color-overlay-text);
  }
</style>
