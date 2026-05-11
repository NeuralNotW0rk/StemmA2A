export const ELEMENT_INFO_CONFIG = {
  // Keys to hide in the ElementInfoView for a cleaner display
  ignoredKeys: new Set([
    'x',
    'y',
    'vx',
    'vy',
    'fx',
    'fy',
    'index',
    'embeddings',
    'value',
    '_incoming_node_ids'
  ]),
  // Attributes to prioritize at the top of the ElementInfoView
  priorityKeys: ['id', 'type', 'name', 'alias']
} as const

export const BATCHING_CONFIG = {
  // Explicit context attributes that must match exactly for nodes to be grouped in a batch.
  strictContextKeys: ['prompt']
} as const