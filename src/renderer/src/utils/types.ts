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
  form_config?: Record<string, unknown>[]
}

export interface OperationInfo {
  name: string
  description?: string
  category?: 'dsp' | 'evolution' | 'generative' | string
  execution?: 'immediate' | 'queued' | string
  execution_mode?: 'sync' | 'async' | string
  initiator_types?: string[]
  output_type?: string
  form_config?: Record<string, unknown>[]
  context_overrides?: Record<string, OperationContextOverride>
}

export type ElementData = Record<string, unknown>

export type NodeFilter = Record<string, unknown>

export type ErrorInfo = { title: string; message: string }
