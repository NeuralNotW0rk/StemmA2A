import type { FormConfig, NodeData } from './forms'

export type ActionPanelView =
  | 'import-model'
  | 'import-grating'
  | 'removal'
  | 'cleanup'
  | 'grouping'
  | 'operation'
  | 'bend'
  | 'none'

export interface OperationContextOverride {
  name?: string
  description?: string
  form_config?: FormConfig
}

export interface OperationInfo {
  name: string
  description?: string
  category?: 'dsp' | 'evolution' | 'generative' | string
  execution?: 'immediate' | 'queued' | string
  execution_mode?: 'sync' | 'async' | string
  initiator_types?: string[]
  /** For a model op shared across adapters: the initiator types each adapter implements it for */
  adapter_initiator_types?: Record<string, string[]>
  output_type?: string
  form_config?: FormConfig
  context_overrides?: Record<string, OperationContextOverride>
}

export type ElementData = Record<string, unknown>

export type NodeFilter = Record<string, unknown>

/** A message for the info panel. `kind` defaults to 'error'. */
export type PanelMessage = { title: string; message: string; kind?: 'error' | 'success' | 'info' }

/** The project a forked project was copied from. */
export interface ProjectOrigin {
  project_name: string
  project_path: string
  element_ids: string[]
  created: number
}

export type EngineMode = 'local' | 'remote'

export interface EngineSettings {
  mode: EngineMode
  // Omitted to reuse the backend's last remote URL
  remote_url?: string
}

export interface EngineState {
  mode: EngineMode
  // Used when switching to remote; kept while the local engine is active
  remote_url: string | null
  unfinished_jobs: number
  // Only present when a check was requested and the remote engine is active
  connection?: EngineConnection
}

export interface EngineConnection {
  reachable: boolean
  // The engine service's device, such as "cuda", when reachable
  device?: string
  error?: string
}

// Returned instead of an EngineState when jobs would be abandoned by the switch
export interface EngineConflict {
  conflict: true
  error: string
  unfinished_jobs: number
}

export interface ForkResult {
  project_name: string
  project_path: string
  element_count: number
  edge_count: number
  copied_files: number
  converted_exemplars: string[]
}

export interface OrphanFile {
  // Relative to the project root, e.g. "generate/foo.wav"
  path: string
  size: number
}

export interface TrashBatch {
  id: string
  created: number | null
  file_count: number
  bytes: number
}

export interface CleanupPreview {
  orphans: OrphanFile[]
  orphan_bytes: number
  trash: TrashBatch[]
  unfinished_jobs: number
}

export interface QuarantineResult {
  message: string
  batch: { id: string; file_count: number; bytes: number } | null
  // Previewed files that were no longer unreferenced, or gone, when the move ran
  skipped: number
}

export interface RestoreResult {
  restored: number
  // Files left in the trash because their original location is occupied again
  conflicts: string[]
}

export interface NodeListItem {
  id: number | string
  node: NodeData | string | null
  // A number normally; a sequence string (e.g. "-1, 0.5, 1..2:0.5") when strengthBatch is on
  strength?: number | string
  strengthBatch?: boolean
  [key: string]: unknown
}

export interface GratingOverride {
  address: string
  kernel_type: string
  targetType: 'all' | 'indices' | 'cluster'
  indicesText: string
  cluster: number
  params: Record<string, number | string | boolean>
  batchFields?: Record<string, boolean>
}

export interface GratingListItem extends NodeListItem {
  loadedNodeId?: string
  overrides?: GratingOverride[]
}

// An exemplar in a target individual's lineage that a preset was found on
export interface ExemplarPresetSource {
  audio_id: string
  audio_name: string
  individual_id: string
  individual_name: string
  // Generations between the exemplar's individual and the nearest target (0 = the target itself)
  distance: number
}

// A distinct set of generation parameters used by exemplars across the target individuals' lineages
export interface ExemplarPreset {
  key: string
  params: Record<string, unknown>
  sources: ExemplarPresetSource[]
  // Target individuals that already have an exemplar generated with these parameters
  covered_ids: string[]
}
