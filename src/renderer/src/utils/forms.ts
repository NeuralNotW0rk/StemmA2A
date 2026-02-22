export interface FormFieldOption {
  label: string;
  value: any;
}

export interface FormField {
  name: string;
  label: string;
  type: 'string' | 'number' | 'boolean' | 'textarea' | 'select';
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
}

export type FormConfig = FormField[];
