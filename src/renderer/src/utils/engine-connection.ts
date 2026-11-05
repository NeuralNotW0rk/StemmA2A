import { derived, writable } from 'svelte/store'
import type { Readable, Writable } from 'svelte/store'
import type { OperationInfo } from './types'

export interface EngineAvailability {
  // The remote engine's URL while it is active; null for the local engine
  remoteUrl: string | null
  // False only once a check has found the remote engine unreachable; null while unknown
  reachable: boolean | null
  error: string | null
}

// Written by the engine menu, which runs the connection checks
export const engineAvailability: Writable<EngineAvailability> = writable({
  remoteUrl: null,
  reachable: null,
  error: null
})

// Why engine operations can't run right now, or null when they can
export const engineUnavailableReason: Readable<string | null> = derived(
  engineAvailability,
  ($a: EngineAvailability): string | null =>
    $a.remoteUrl && $a.reachable === false
      ? `The remote engine (${$a.remoteUrl}) can't be reached, so operations that run models are unavailable. ` +
        'Check that the engine service is running and that this machine is online, or switch to the local engine from the Engine menu.'
      : null
)

// Operations that run on the engine, rather than on this machine's backend
export function runsOnEngine(op: Pick<OperationInfo, 'category' | 'execution_mode'> | null): boolean {
  if (!op) return false
  return op.category === 'generative' || (!op.category && op.execution_mode === 'async')
}
