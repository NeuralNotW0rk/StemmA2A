import { writable } from 'svelte/store';
import type { ModelData, AudioData } from './forms';

type SelectableNodeData = ModelData | AudioData;

interface SelectionState {
  isSelecting: boolean;
  selectionType: 'model' | 'audio' | null;
  boundNodeId: string | null;
  onSelect: ((node: SelectableNodeData) => void) | null;
}

function createSelectionStore() {
  const { subscribe, set, update } = writable<SelectionState>({
    isSelecting: false,
    selectionType: null,
    boundNodeId: null,
    onSelect: null,
  });

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
    },
  };
}

export const selectionStore = createSelectionStore();

export const activeNodeStore = writable<any>(null);
