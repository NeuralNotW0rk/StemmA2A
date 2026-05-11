
export type ActionPanelView = 'generation' | 'import-model' | 'import-lattice' | 'removal' | 'batching' | 'none'

export type ElementData = Record<string, unknown>

export type NodeFilter = Record<string, unknown>

export type ErrorInfo = { title: string; message: string }
