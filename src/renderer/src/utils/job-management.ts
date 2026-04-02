import { writable } from 'svelte/store'
import type { Writable } from 'svelte/store'
import { v4 as uuidv4 } from 'uuid'

export type JobStatus = 'running' | 'success' | 'error' | 'cancelled'

export type JobResult = Record<string, any> & { viewed?: boolean }

export interface Job {
  id: string
  name: string
  status: JobStatus
  payload: unknown
  result: JobResult | null
  error: { title: string; message: string } | null
  createdAt: number
  updatedAt: number
}

export const jobStore: Writable<Job[]> = writable([])

export function addJob(name: string, payload: unknown): Job {
  const job: Job = {
    id: uuidv4(),
    name,
    status: 'running',
    payload,
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

export function removeJob(jobId: string): void {
  jobStore.update((jobs) => jobs.filter((j) => j.id !== jobId))
}

export function clearJobs(): void {
  jobStore.set([])
}
