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
  output_type?: string
  form_config?: FormConfig
  context_overrides?: Record<string, OperationContextOverride>
}

export type ElementData = Record<string, unknown>

export type NodeFilter = Record<string, unknown>

export type ErrorInfo = { title: string; message: string }

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
