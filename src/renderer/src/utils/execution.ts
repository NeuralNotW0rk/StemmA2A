import { writable } from 'svelte/store'
import type { Writable } from 'svelte/store'
import { addJob, updateJob, pollJobStatus } from './job-management'

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
    const payloadWithId =
      typeof payload === 'object' && payload !== null
        ? { ...payload, job_id: job.id }
        : { job_id: job.id, payload }
    
    const initData = await window.api.generate(payloadWithId)
    
    if (initData.status === 'running') {
      const finalResult = await pollJobStatus(initData.job_id)
      job.status = 'success'
      job.result = finalResult
      updateJob(job)
    } else {
      job.status = 'success'
      job.result = initData
      updateJob(job)
    }
  } catch (e: unknown) {
    let message = e instanceof Error ? e.message : String(e)
    message = message.replace(/^Error invoking remote method '[^']+':\s*(Error:\s*)?/, '')
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
    let message = e instanceof Error ? e.message : String(e)
    message = message.replace(/^Error invoking remote method '[^']+':\s*(Error:\s*)?/, '')
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
