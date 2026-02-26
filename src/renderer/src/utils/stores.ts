import { writable } from 'svelte/store';
import type { ModelData, AudioData } from './forms';

type SelectableNodeData = ModelData | AudioData;

interface SelectionState {
  isSelecting: boolean;
  selectionType: 'model' | 'audio' | null;
  onSelect: ((node: SelectableNodeData) => void) | null;
}

function createSelectionStore() {
  const { subscribe, set, update } = writable<SelectionState>({
    isSelecting: false,
    selectionType: null,
    onSelect: null,
  });

  return {
    subscribe,
    startSelection: (selectionType: 'model' | 'audio', onSelect: (node: SelectableNodeData) => void) =>
      update((state) => ({
        ...state,
        isSelecting: true,
        selectionType,
        onSelect,
      })),
    cancelSelection: () =>
      update((state) => ({
        ...state,
        isSelecting: false,
        selectionType: null,
        onSelect: null,
      })),
    resolveSelection: (node: SelectableNodeData) => {
      update((state) => {
        if (state.isSelecting && state.onSelect) {
          state.onSelect(node);
        }
        return {
          ...state,
          isSelecting: false,
          selectionType: null,
          onSelect: null,
        };
      });
    },
  };
}

export const selectionStore = createSelectionStore();
