import { writable } from 'svelte/store'
import type { Writable } from 'svelte/store'

export type ExecutionStatus = 'idle' | 'running' | 'success' | 'error'

export interface ExecutionState {
  status: ExecutionStatus
  payload: unknown | null
  result: unknown | null
  error: { title: string; message: string } | null
}

export const executionStore: Writable<ExecutionState> = writable({
  status: 'idle',
  payload: null,
  result: null,
  error: null
})

export const embeddingUpdateExecutionStore: Writable<ExecutionState> = writable({
  status: 'idle',
  payload: null,
  result: null,
  error: null
})

export async function startExecution(payload: unknown): Promise<void> {
  executionStore.set({
    status: 'running',
    payload,
    result: null,
    error: null
  })

  try {
    const result = await window.api.generate(payload)
    executionStore.set({
      status: 'success',
      payload,
      result,
      error: null
    })
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e)
    const error = { title: 'Generation Failed', message }
    executionStore.set({
      status: 'error',
      payload,
      result: null,
      error
    })
  }
}

export async function startEmbeddingUpdate(): Promise<void> {
  embeddingUpdateExecutionStore.set({
    status: 'running',
    payload: null,
    result: null,
    error: null
  })

  try {
    const result = await window.api.updateEmbeddings()
    embeddingUpdateExecutionStore.set({
      status: 'success',
      payload: null,
      result,
      error: null
    })
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e)
    const error = { title: 'Embedding Update Failed', message }
    embeddingUpdateExecutionStore.set({
      status: 'error',
      payload: null,
      result: null,
      error
    })
  }
}

export function clearExecutionStore(store: Writable<ExecutionState>): void {
  store.set({
    status: 'idle',
    payload: null,
    result: null,
    error: null
  })
}
