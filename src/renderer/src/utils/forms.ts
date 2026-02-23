export interface FormFieldOption {
  label: string;
  value: any;
}

export interface FormField {
  name: string;
  label:string;
  type: 'string' | 'number' | 'boolean' | 'textarea' | 'select' | 'file';
  defaultValue?: any;
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
}

export type FormConfig = FormField[];
