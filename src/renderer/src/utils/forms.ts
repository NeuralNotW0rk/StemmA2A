export interface FormFieldOption {
  label: string;
  value: any;
}

export interface FormField {
  name: string;
  label:string;
  type: 'string' | 'number' | 'boolean' | 'textarea' | 'select' | 'file' | 'node';
  defaultValue?: any;
  selectionType?: 'model' | 'audio';
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
  initiatorNode: NodeData | null
): { formData: Record<string, unknown>; boundNodes: Record<string, NodeData> } {
  const formData: Record<string, unknown> = {}
  const boundNodes: Record<string, NodeData> = {}

  if (!fields) {
    return { formData, boundNodes }
  }

  for (const field of fields) {
    if (field.defaultValue !== undefined) {
      formData[field.name] = field.defaultValue
    }
    // Check if the initiator node can be used for a field
    if (
      initiatorNode &&
      field.type === 'node' &&
      field.selectionType === initiatorNode.type
    ) {
      formData[field.name] = initiatorNode
      boundNodes[field.name] = initiatorNode
    }
  }
  return { formData, boundNodes }
}
