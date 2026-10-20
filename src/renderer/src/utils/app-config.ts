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
    'value',
    'job_id',
    'execution_mode'
  ]),
  // Attributes to prioritize at the top of the ElementInfoView
  priorityKeys: ['id', 'type', 'name', 'alias'],
  // Fields start collapsed when an array/object has more items than this...
  collapseItemThreshold: 8,
  // ...or when a primitive's serialized length exceeds this many characters
  collapseCharThreshold: 200,
  // Max characters of a collapsed primitive shown in its header preview
  previewLength: 60
} as const

export const GROUPING_CONFIG = {
  // Explicit context attributes that must match exactly for nodes to be grouped in a group.
  strictContextKeys: ['prompt']
} as const