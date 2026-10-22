import type { FormConfig, NodeData } from './forms'

export type ActionPanelView =
  | 'import-model'
  | 'import-grating'
  | 'removal'
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

export interface ForkResult {
  project_name: string
  project_path: string
  element_count: number
  edge_count: number
  copied_files: number
  converted_exemplars: string[]
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
