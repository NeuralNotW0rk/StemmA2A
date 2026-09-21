import type { NodeFilter } from './types'

export interface FormFieldOption {
  label: string;
  value: any;
}

export interface FormField {
  name: string;
  label: string;
  type: 'string' | 'number' | 'boolean' | 'textarea' | 'select' | 'file' | 'directory' | 'node' | 'node-list';
  defaultValue?: any;
  filter?: NodeFilter;
  options?: FormFieldOption[];
  placeholder?: string;
  minItems?: number;
  itemLabel?: string;
  validation?: {
    required?: boolean;
    min?: number;
    max?: number;
    minLength?: number;
    maxLength?: number;
  };
  filters?: { name: string; extensions: string[] }[];
  show_if?: Record<string, any> | Record<string, any>[];
  conditionalDefaults?: {
    show_if: Record<string, any>;
    value: any;
  }[];
  allowSequence?: boolean;
}

export type FormConfig = FormField[];

export interface ModelData {
  id: string
  name: string
  type: 'model'
  adapter: string
  model_type?: string
  alias?: string
  [key: string]: unknown
}

export interface AudioData {
  id: string
  name: string
  type: 'audio'
  alias?: string
  [key: string]: unknown
}

export interface GroupData {
  id: string
  name?: string
  type: 'group'
  member_type: string
  member_ids: string[]
  alias?: string
  [key: string]: unknown
}

export interface BundleData {
  id: string
  name: string
  type: 'bundle'
  member_type: string
  member_ids: string[]
  alias?: string
  context: Record<string, unknown>
  [key: string]: unknown
}

export interface GratingElementMetadata {
  indices?: number[]
  cluster?: number | null
  cluster_map?: unknown
  radius?: number
  scale_factor?: number
  angle?: number
  offset_x?: number
  offset_y?: number
  factor?: number
  multiplier?: number
  scale_x?: number
  scale_y?: number
  threshold?: number
  [key: string]: unknown
}

export interface GratingElement {
  address: string
  kernel_type: string
  metadata?: GratingElementMetadata
  params?: Record<string, unknown>
  indices?: number[]
  perform_clustering?: boolean
  num_clusters?: number | null
  cluster?: number | null
  [key: string]: unknown
}

export interface GratingData {
  id: string
  name?: string
  type: 'grating'
  elements?: GratingElement[]
  base_model_id?: string
  alias?: string
  [key: string]: unknown
}

export type NodeData = ModelData | AudioData | GroupData | BundleData | GratingData

export function initializeFormData(
  fields: FormConfig,
  context: Record<string, any> | null,
  initiatorNode: Record<string, any> | null = null,
  currentData: Record<string, any> | null = null
): { formData: Record<string, unknown> } {
  const formData: Record<string, unknown> = currentData ? { ...currentData } : {}

  if (!fields) {
    return { formData }
  }

  for (const field of fields) {
    if (context) {
      if (context[field.name] !== undefined) {
        formData[field.name] = context[field.name]
      } else if (field.name === 'cfg_scale') {
        if (context.inversion_cfg_scale !== undefined) {
          formData[field.name] = context.inversion_cfg_scale
        } else if (context.inversion_metadata?.inversion_cfg_scale !== undefined) {
          formData[field.name] = context.inversion_metadata.inversion_cfg_scale
        }
      } else if (field.name === 'inversion_cfg_scale') {
        if (context.cfg_scale !== undefined) {
          formData[field.name] = context.cfg_scale
        } else if (context.inversion_metadata?.inversion_cfg_scale !== undefined) {
          formData[field.name] = context.inversion_metadata.inversion_cfg_scale
        }
      } else if (field.type === 'node' && context[`${field.name}_id`] !== undefined) {
        formData[field.name] = context[`${field.name}_id`]
      }
    }
    if (
      formData[field.name] === undefined &&
      initiatorNode &&
      field.type === 'node' &&
      field.filter
    ) {
      if (field.filter.type === initiatorNode.type) {
        formData[field.name] = initiatorNode
      } else if (
        initiatorNode.type === 'group' &&
        field.filter.type === initiatorNode.member_type
      ) {
        formData[field.name] = initiatorNode
      }
    }
    if (formData[field.name] === undefined && field.type === 'node-list') {
      const listItems: Array<{ id: number; node: any }> = []
      if (initiatorNode) {
        if (Array.isArray(initiatorNode.selectedNodes) && initiatorNode.selectedNodes.length > 0) {
          initiatorNode.selectedNodes.forEach((n: any, idx: number) => {
            if (n && !n.source && !n.target) {
              if (!field.filter || !field.filter.type || n.type === field.filter.type) {
                listItems.push({ id: idx, node: n })
              }
            }
          })
        } else if (initiatorNode.type === 'group' && Array.isArray(initiatorNode.member_ids) && initiatorNode.member_ids.length > 0) {
          initiatorNode.member_ids.forEach((mId: string, idx: number) => {
            if (typeof mId === 'string' && !mId.includes('->')) {
              listItems.push({ id: idx, node: mId })
            }
          })
        } else if (initiatorNode.type === 'group' && Array.isArray(initiatorNode.members) && initiatorNode.members.length > 0) {
          initiatorNode.members.forEach((n: any, idx: number) => {
            if (n && !n.source && !n.target) {
              if (!field.filter || !field.filter.type || n.type === field.filter.type) {
                listItems.push({ id: idx, node: n })
              }
            }
          })
        } else if (!initiatorNode.source && !initiatorNode.target && (!field.filter || !field.filter.type || initiatorNode.type === field.filter.type)) {
          listItems.push({ id: 0, node: initiatorNode })
        }
      }
      formData[field.name] = listItems
    }
    if (formData[field.name] === undefined) {
      let val = field.defaultValue;
      let hasLatentContextValue = false;
      
      if (initiatorNode && initiatorNode.type === 'latent' && initiatorNode.context) {
        const latCtx = initiatorNode.context;
        if (field.name === 'prompt' && latCtx.prompt !== undefined) {
          val = latCtx.prompt;
          hasLatentContextValue = true;
        } else if (field.name === 'steps' && latCtx.steps !== undefined) {
          val = latCtx.steps;
          hasLatentContextValue = true;
        } else if (field.name === 'seed' && latCtx.seed !== undefined) {
          val = latCtx.seed;
          hasLatentContextValue = true;
        } else if (field.name === 'sigma_min' && latCtx.sigma_min !== undefined) {
          val = latCtx.sigma_min;
          hasLatentContextValue = true;
        } else if (field.name === 'sigma_max' && latCtx.sigma_max !== undefined) {
          val = latCtx.sigma_max;
          hasLatentContextValue = true;
        } else if (field.name === 'cfg_scale' || field.name === 'inversion_cfg_scale') {
          if (latCtx.inversion_cfg_scale !== undefined) {
            val = latCtx.inversion_cfg_scale;
            hasLatentContextValue = true;
          } else if (latCtx.inversion_metadata?.inversion_cfg_scale !== undefined) {
            val = latCtx.inversion_metadata.inversion_cfg_scale;
            hasLatentContextValue = true;
          } else if (latCtx.cfg_scale !== undefined) {
            val = latCtx.cfg_scale;
            hasLatentContextValue = true;
          }
        }
      }

      if (!hasLatentContextValue && field.conditionalDefaults) {
        const data = { ...context, ...formData };
        for (const condDefault of field.conditionalDefaults) {
          let conditionsMet = true;
          for (const key in condDefault.show_if) {
            const condition = condDefault.show_if[key];
            if (condition === 'exists') {
              if (!data[key]) {
                conditionsMet = false;
                break;
              }
            } else {
              if (data[key] !== condition) {
                conditionsMet = false;
                break;
              }
            }
          }
          if (conditionsMet) {
            val = condDefault.value;
            break;
          }
        }
      }
      if (field.name === 'seconds_total' && initiatorNode) {
        if (initiatorNode.type === 'audio' && initiatorNode.duration !== undefined) {
          formData[field.name] = initiatorNode.duration;
        } else if (initiatorNode.type === 'latent') {
          const duration = initiatorNode.duration || initiatorNode.context?.seconds_total || initiatorNode.context?.inversion_metadata?.seconds_total;
          if (duration !== undefined) {
            formData[field.name] = duration;
          } else if (val !== undefined) {
            formData[field.name] = val;
          }
        } else if (val !== undefined) {
          formData[field.name] = val;
        }
      } else if (val !== undefined) {
        formData[field.name] = val;
      }
    }
  }
  return { formData }
}
