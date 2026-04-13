import { writable, get } from 'svelte/store'
import type { Writable } from 'svelte/store'
import { v4 as uuidv4 } from 'uuid'
import { cyInstanceStore } from './stores'

export type JobStatus = 'pending' | 'running' | 'success' | 'error' | 'cancelled' | 'cancelling'

export type JobResult = Record<string, any> & { viewed?: boolean }

export interface JobProgress {
  value: number
  total: number
  description?: string
}

export interface Job {
  id: string
  name: string
  status: JobStatus
  payload: unknown
  progress: JobProgress | null
  result: JobResult | null
  error: { title: string; message: string } | null
  createdAt: number
  updatedAt: number
}

export const jobStore: Writable<Job[]> = writable([])

export function addJob(name: string, payload: unknown, initialStatus: JobStatus = 'running'): Job {
  const job: Job = {
    id: uuidv4(),
    name,
    status: initialStatus,
    payload,
    progress: null,
    result: null,
    error: null,
    createdAt: Date.now(),
    updatedAt: Date.now()
  }

  jobStore.update((jobs) => [...jobs, job])
  return job
}

export function updateJob(job: Job): void {
  job.updatedAt = Date.now()
  jobStore.update((jobs) => {
    const index = jobs.findIndex((j) => j.id === job.id)
    if (index !== -1) {
      jobs[index] = job
    }
    return jobs
  })
}

export function updateJobProgress(jobId: string, progress: JobProgress): void {
  jobStore.update((jobs) => {
    const job = jobs.find((j) => j.id === jobId)
    if (job && (job.status === 'running' || job.status === 'pending')) {
      job.progress = progress
      job.updatedAt = Date.now()
    }
    return jobs
  })
}

export function removeJob(jobId: string): void {
  jobStore.update((jobs) => jobs.filter((j) => j.id !== jobId))
}

export async function cancelJob(jobId: string): Promise<void> {
  const jobs = get(jobStore)
  const job = jobs.find((j) => j.id === jobId)
  if (job) {
    job.status = 'cancelling'
    updateJob(job)
    try {
      await window.api.cancelJob(jobId)
      job.status = 'cancelled'
      updateJob(job)
    } catch (error) {
      job.status = 'running' // Or 'error'
      updateJob(job)
      console.error('Error cancelling job:', error)
    }
  }
}

export function clearJobs(): void {
  jobStore.set([])
}

export function focusJobNode(job: Job): void {
  if (job.status !== 'success' || !job.result) return

  // Replace `node_id` or `id` with whatever property your backend
  // actually uses to return the newly created node's ID.
  const nodeId = job.result.node_id || job.result.id

  if (!nodeId) {
    console.warn('No node ID found in job result to focus on.')
    return
  }

  const cy = get(cyInstanceStore)
  if (cy) {
    const node = cy.getElementById(nodeId)
    if (node.length > 0) {
      cy.elements().unselect()
      node.select()

      cy.animate({
        center: { eles: node },
        zoom: Math.max(cy.zoom(), 1.5), // Ensure a reasonable zoom level
        duration: 500
      })
    } else {
      console.warn(`Node with ID ${nodeId} not found in the graph.`)
    }
  }
}

export async function pollJobStatus(jobId: string, intervalMs = 1500): Promise<any> {
  while (true) {
    // Use the IPC bridge instead of a direct HTTP fetch
    const data = await window.api.pollJobStatus(jobId)
    
    if (data.status === 'completed') {
      return data // Returns the fully processed artifact and context
    } else if (data.status === 'failed' || data.status === 'not_found' || data.error) {
      let errMsg = data.error || 'Job failed'
      if (data.traceback) {
        errMsg += `\n\nTraceback:\n${data.traceback}`
      }
      throw new Error(errMsg)
    } else if (data.status === 'running' || data.status === 'pending') {
      // Update the job store with the intermediate status to reflect in the UI
      jobStore.update((jobs) => {
        const job = jobs.find((j) => j.id === jobId)
        if (job && job.status !== data.status) {
          job.status = data.status
          job.updatedAt = Date.now()
        }
        return jobs
      })
    }
    
    // Wait for the next interval before asking the backend again
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
}
