import { writable } from 'svelte/store'
import type { Core } from 'cytoscape'
import type { ModelData, NodeData } from './forms'

export const cyInstanceStore = writable<Core | null>(null);

type SelectableNodeData = NodeData

interface SelectionState {
  isSelecting: boolean
  filter: Record<string, string | number | boolean> | null
  boundNodeId: string | null
  onSelect: ((node: SelectableNodeData) => void) | null
}


function createSelectionStore() {
  const { subscribe, update } = writable<SelectionState>({
    isSelecting: false,
    filter: null,
    boundNodeId: null,
    onSelect: null
  })

  return {
    subscribe,
    startSelection: (
      filter: Record<string, string | number | boolean>,
      boundNodeId: string | null,
      onSelect: (node: SelectableNodeData) => void
    ) =>
      update((state) => ({
        ...state,
        isSelecting: true,
        filter,
        boundNodeId,
        onSelect
      })),
    cancelSelection: () =>
      update((state) => ({
        ...state,
        isSelecting: false,
        filter: null,
        boundNodeId: null,
        onSelect: null
      })),
    resolveSelection: (node: SelectableNodeData) => {
      let onSelectCallback: ((node: SelectableNodeData) => void) | null = null

      update((state) => {
        if (state.isSelecting && state.onSelect) {
          onSelectCallback = state.onSelect
        }
        return {
          ...state,
          isSelecting: false,
          filter: null,
          boundNodeId: null,
          onSelect: null
        }
      })

      if (onSelectCallback) {
        onSelectCallback(node)
      }
    }
  }
}

export const selectionStore = createSelectionStore()

// A minimal type for graph elements, to be expanded as needed.
export interface GraphElement {
  id: string
  type: string
  [key: string]: any
}

export const initiatorNodeStore = writable<GraphElement | null>(null)
export const contextStore = writable<Record<string, any> | null>(null)

export const lastUsedModelStore = writable<ModelData | null>(null)

// Types for content panel views
export type View = 'element-info' | 'error' | 'generation' | 'import-model' | 'removal'

export type BackendStatus = {
  project_loaded: boolean
  device: string
  server_instance_id?: string | null
}

export const backendStatus = writable<BackendStatus>({
  project_loaded: false,
  device: 'unknown'
})

export const activeView = writable<View>('element-info')
export const selectedForRemoval = writable<GraphElement | null>(null)

export const isCreatingNewProject = writable<boolean>(false)
export const newProjectName = writable<string>('')

// ===============================================
// Form State Store
// Holds state for ephemeral form interactions,
// like selecting nodes for a form field.
// ===============================================

interface FormState {
  generationModel: ModelData | null
}

function createFormStateStore() {
  const { subscribe, set, update } = writable<FormState>({
    generationModel: null
  })

  return {
    subscribe,
    set,
    update,
    clearGenerationModel: () => {
      update((state) => ({ ...state, generationModel: null }))
    }
  }
}

export const formStateStore = createFormStateStore()
