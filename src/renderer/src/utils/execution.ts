import { writable } from 'svelte/store'
import type { Writable } from 'svelte/store'
import { addJob, updateJob } from './job-management'

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

export async function startExecution(name: string, payload: unknown): Promise<void> {
  const job = addJob(name, payload)

  try {
    const result = await window.api.generate(payload)
    job.status = 'success'
    job.result = result
    updateJob(job)
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e)
    const error = { title: 'Generation Failed', message }
    job.status = 'error'
    job.error = error
    updateJob(job)
  }
}

export async function startEmbeddingUpdate(): Promise<void> {
  const job = addJob('Updating Embeddings', {})

  try {
    const result = await window.api.updateEmbeddings()
    job.status = 'success'
    job.result = result
    updateJob(job)
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e)
    const error = { title: 'Embedding Update Failed', message }
    job.status = 'error'
    job.error = error
    updateJob(job)
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
