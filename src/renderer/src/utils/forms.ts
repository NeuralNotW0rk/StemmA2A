import type { NodeFilter } from './types'

export interface FormFieldOption {
  label: string;
  value: any;
}

export interface FormField {
  name: string;
  label:string;
  type: 'string' | 'number' | 'boolean' | 'textarea' | 'select' | 'file' | 'directory' | 'node';
  defaultValue?: any;
  filter?: NodeFilter;
  options?: FormFieldOption[];
  placeholder?: string;
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

export interface BatchData {
  id: string
  name?: string
  type: 'batch'
  member_type: string
  member_ids: string[]
  alias?: string
  [key: string]: unknown
}

export type NodeData = ModelData | AudioData | BatchData

export function initializeFormData(
  fields: FormConfig,
  context: Record<string, any> | null,
  initiatorNode: Record<string, any> | null = null
): { formData: Record<string, unknown> } {
  const formData: Record<string, unknown> = {}

  if (!fields) {
    return { formData }
  }

  for (const field of fields) {
    if (context) {
      if (context[field.name] !== undefined) {
        formData[field.name] = context[field.name]
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
        initiatorNode.type === 'batch' &&
        field.filter.type === initiatorNode.member_type
      ) {
        formData[field.name] = initiatorNode
      }
    }
    if (formData[field.name] === undefined) {
      let val = field.defaultValue;
      if (field.conditionalDefaults) {
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
      if (field.name === 'seconds_total' && initiatorNode && initiatorNode.type === 'audio' && initiatorNode.duration !== undefined) {
        formData[field.name] = initiatorNode.duration;
      } else if (val !== undefined) {
        formData[field.name] = val;
      }
    }
  }
  return { formData }
}
