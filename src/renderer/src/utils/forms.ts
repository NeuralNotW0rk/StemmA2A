export interface FormFieldOption {
  label: string;
  value: any;
}

export interface FormField {
  name: string;
  label:string;
  type: 'string' | 'number' | 'boolean' | 'textarea' | 'select' | 'file' | 'node';
  defaultValue?: any;
  filter?: Record<string, string | number | boolean>;
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
  show_if?: Record<string, any>;
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

export type NodeData = ModelData | AudioData

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
      field.filter &&
      field.filter.type === initiatorNode.type
    ) {
      formData[field.name] = initiatorNode
    }
    if (formData[field.name] === undefined && field.defaultValue !== undefined) {
      formData[field.name] = field.defaultValue
    }
  }
  return { formData }
}
