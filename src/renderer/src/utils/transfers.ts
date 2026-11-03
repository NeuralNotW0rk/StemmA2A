import { writable, get } from 'svelte/store'
import type { Writable } from 'svelte/store'
import { jobStore } from './job-management'

// Mirrors _TransferTracker.to_dict() in backend/engine/remote_engine.py
export interface Transfer {
  id: string
  direction: 'upload' | 'download'
  uid: string
  name: string
  state: 'queued' | 'active' | 'done' | 'failed'
  bytes_done: number
  total_bytes: number
  files_done: number
  total_files: number
  attempt: number
  max_attempts: number
  bytes_per_second: number
  error: string | null
  job_ids: string[]
  started_at: number
  active_at: number | null
  finished_at: number | null
}

// Below both of these a transfer is too quick to be worth listing; it only clutters the Jobs panel
const SLOW_TRANSFER_SECONDS = 2
const LARGE_TRANSFER_BYTES = 100 * 1024 * 1024

/**
 * Whether a transfer is worth listing: a failed one, a large one, or one that has spent a while
 * moving data. Time spent queued for an upload slot doesn't count, or every file of a batch would
 * qualify by waiting its turn. The host stamps the times on this machine, so they compare to Date.now().
 */
export function isNotableTransfer(transfer: Transfer): boolean {
  if (transfer.state === 'failed' || transfer.total_bytes >= LARGE_TRANSFER_BYTES) return true
  if (transfer.active_at === null) return false
  const end = transfer.finished_at ?? Date.now() / 1000
  return end - transfer.active_at >= SLOW_TRANSFER_SECONDS
}

export const transferStore: Writable<Transfer[]> = writable([])

const POLL_INTERVAL_MS = 1000
let pollTimer: ReturnType<typeof setInterval> | null = null

async function refreshTransfers(): Promise<void> {
  try {
    const data = (await window.api.getTransfers()) as { transfers?: Transfer[] }
    transferStore.set(data.transfers ?? [])
  } catch {
    // The backend may be restarting; the next poll tries again
  }
}

function hasActiveJobs(): boolean {
  return get(jobStore).some((job) => job.status === 'pending' || job.status === 'running')
}

/**
 * Polls the backend for transfers while any job is pending or running, since transfers only happen
 * on a job's behalf. Once the jobs settle it keeps polling until the finished transfers drop off,
 * so each one is seen to end. Call once at startup.
 */
export function startTransferPolling(): () => void {
  const unsubscribe = jobStore.subscribe(() => {
    if (pollTimer || !hasActiveJobs()) return
    refreshTransfers()
    pollTimer = setInterval(() => {
      if (!hasActiveJobs() && get(transferStore).length === 0) {
        clearInterval(pollTimer!)
        pollTimer = null
        return
      }
      refreshTransfers()
    }, POLL_INTERVAL_MS)
  })
  return () => {
    unsubscribe()
    if (pollTimer) clearInterval(pollTimer)
    pollTimer = null
  }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit++
  }
  return `${value.toFixed(value < 10 ? 2 : 1)} ${units[unit]}`
}

export function transferPercent(transfer: Transfer): number | null {
  if (transfer.state === 'done') return 100
  if (!transfer.total_bytes) return null
  return Math.min(100, (transfer.bytes_done / transfer.total_bytes) * 100)
}

/**
 * One line for the transfers a job is waiting on, e.g. "Uploading model.ckpt: 42%" or, for a batch,
 * "Uploading 8 files" (the count falls as they finish).
 */
export function jobTransferSummary(transfers: Transfer[]): string | null {
  if (transfers.length === 0) return null
  const allQueued = transfers.every((t) => t.state === 'queued')
  const directions = new Set(transfers.map((t) => t.direction))
  const verb = allQueued
    ? 'Waiting to upload'
    : directions.size > 1
      ? 'Transferring'
      : directions.has('upload')
        ? 'Uploading'
        : 'Downloading'

  if (transfers.length > 1) return `${verb} ${transfers.length} files`

  const [transfer] = transfers
  const percent = transferPercent(transfer)
  return `${verb} ${transfer.name}${percent !== null && transfer.state === 'active' ? `: ${Math.floor(percent)}%` : ''}`
}

/** One-line status, e.g. "1.20 / 3.40 GB · 5.1 MB/s". */
export function transferDetail(transfer: Transfer): string {
  switch (transfer.state) {
    case 'queued':
      return 'Waiting for an upload slot'
    case 'failed':
      return transfer.error || 'Failed'
    case 'done':
      return `${formatBytes(transfer.total_bytes || transfer.bytes_done)} · done`
  }
  const parts = [
    transfer.total_bytes
      ? `${formatBytes(transfer.bytes_done)} / ${formatBytes(transfer.total_bytes)}`
      : formatBytes(transfer.bytes_done)
  ]
  if (transfer.bytes_per_second > 0) parts.push(`${formatBytes(transfer.bytes_per_second)}/s`)
  if (transfer.total_files > 1) parts.push(`file ${Math.min(transfer.files_done + 1, transfer.total_files)} of ${transfer.total_files}`)
  if (transfer.attempt > 1) parts.push(`attempt ${transfer.attempt}/${transfer.max_attempts}`)
  return parts.join(' · ')
}
