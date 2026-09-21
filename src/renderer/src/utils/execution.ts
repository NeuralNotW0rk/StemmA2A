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

export async function startExecution(
  name: string, 
  payload: unknown, 
  operation: string = 'generate',
  executionMode?: 'sync' | 'async'
): Promise<void> {
  const job = addJob(name, payload, 'pending')

  try {
    const payloadWithId =
      typeof payload === 'object' && payload !== null
        ? { ...payload, job_id: job.id, operation, ...(executionMode ? { execution_mode: executionMode } : {}) }
        : { job_id: job.id, payload, operation, ...(executionMode ? { execution_mode: executionMode } : {}) }
    
    const initData = await window.api.executeOperation(payloadWithId)
    
    if (initData.status === 'running' || initData.status === 'pending') {
      const finalResult = await pollJobStatus(initData.job_id)
      job.status = 'success'
      job.result = finalResult
      updateJob(job)

      // If this was a compound evolution job that queued child generation jobs, poll all sub-jobs
      if (finalResult && typeof finalResult === 'object') {
        const res = ((finalResult as Record<string, unknown>).result || finalResult) as Record<string, unknown>
        if (Array.isArray(res.job_ids) && res.job_ids.length > 0) {
          const totalSubJobs = res.job_ids.length
          res.job_ids.forEach((subJobId: unknown, idx: number) => {
            if (typeof subJobId === 'string') {
              addJob(`Exemplar Generation (${idx + 1}/${totalSubJobs})`, null, 'running', subJobId)
              pollJobStatus(subJobId).catch((subErr: unknown) => {
                console.error('Sub-job polling error:', subErr)
              })
            }
          })
        }
      }
    } else {
      job.status = 'success'
      job.result = initData
      updateJob(job)

      if (initData && typeof initData === 'object') {
        const res = ((initData as Record<string, unknown>).result || initData) as Record<string, unknown>
        if (Array.isArray(res.job_ids) && res.job_ids.length > 0) {
          const totalSubJobs = res.job_ids.length
          res.job_ids.forEach((subJobId: unknown, idx: number) => {
            if (typeof subJobId === 'string') {
              addJob(`Exemplar Generation (${idx + 1}/${totalSubJobs})`, null, 'running', subJobId)
              pollJobStatus(subJobId).catch((subErr: unknown) => {
                console.error('Sub-job polling error:', subErr)
              })
            }
          })
        }
      }
    }
  } catch (e: unknown) {
    let message = e instanceof Error ? e.message : String(e)
    message = message.replace(/^Error invoking remote method '[^']+':\s*(Error:\s*)?/, '')
    const error = { title: 'Execution Failed', message }
    job.status = 'error'
    job.error = error
    updateJob(job)
  }
}

export async function startEmbeddingUpdate(): Promise<void> {
  try {
    const res = (await window.api.updateEmbeddings()) as { success?: boolean; job_id?: string }
    if (res?.job_id) {
      addJob('Updating Embeddings', {}, 'running', res.job_id)
      await pollJobStatus(res.job_id)
    }
  } catch (e: unknown) {
    let message = e instanceof Error ? e.message : String(e)
    message = message.replace(/^Error invoking remote method '[^']+':\s*(Error:\s*)?/, '')
    throw new Error(message)
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
