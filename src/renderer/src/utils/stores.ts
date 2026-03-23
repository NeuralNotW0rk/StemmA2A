import { writable } from 'svelte/store'
import type { ModelData, AudioData, NodeData } from './forms'

type SelectableNodeData = NodeData

interface SelectionState {
  isSelecting: boolean
  selectionType: 'model' | 'audio' | null
  boundNodeId: string | null
  onSelect: ((node: SelectableNodeData) => void) | null
}

function createSelectionStore() {
  const { subscribe, set, update } = writable<SelectionState>({
    isSelecting: false,
    selectionType: null,
    boundNodeId: null,
    onSelect: null
  })

  return {
    subscribe,
    startSelection: (
      selectionType: 'model' | 'audio',
      boundNodeId: string | null,
      onSelect: (node: SelectableNodeData) => void
    ) =>
      update((state) => ({
        ...state,
        isSelecting: true,
        selectionType,
        boundNodeId,
        onSelect
      })),
    cancelSelection: () =>
      update((state) => ({
        ...state,
        isSelecting: false,
        selectionType: null,
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
          selectionType: null,
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

export const initiatorNodeStore = writable<NodeData | null>(null)

export interface ActiveNodes {
  [role:string]: GraphElement | null
}
export const boundNodeStore = writable<ActiveNodes>({})

export const lastUsedModelStore = writable<ModelData | null>(null)

// Types for content panel views
export type View = 'element-info' | 'error' | 'generation' | 'import-model' | 'removal'

export type BackendStatus = {
  project_loaded: boolean
  device: string
}

export const backendStatus = writable<BackendStatus>({
  project_loaded: false,
  device: 'unknown'
})

export const activeView = writable<View>('element-info')
export const selectedForRemoval = writable<GraphElement | null>(null)

export const isCreatingNewProject = writable<boolean>(false)
export const newProjectName = writable<string>('')
