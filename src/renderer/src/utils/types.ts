
export type ActionPanelView = 'import-model' | 'import-grating' | 'removal' | 'batching' | 'operation' | 'none'

export type ElementData = Record<string, unknown>

export type NodeFilter = Record<string, unknown>

export type ErrorInfo = { title: string; message: string }
