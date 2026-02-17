export interface EngineField {
  key: string
  label: string
  type: 'file' | 'text'
  filters?: { name: string; extensions: string[] }[]
  placeholder?: string
  required?: boolean
}

export interface EngineConfig {
  id: string
  name: string
  description: string
  fields: EngineField[]
}

export const engines: EngineConfig[] = [
  {
    id: 'default',
    name: 'Default',
    description: 'Automatically download and use the latest version of Stable Audio Open Small (uses the Stable Audio Tools engine)',
    fields: []
  },
  {
    id: 'stable-audio-tools',
    name: 'Stable Audio Tools',
    description: 'Manually choose any Stable Audio-based model checkpoint and config file',
    fields: [
      {
        key: 'checkpoint_path',
        label: 'Checkpoint Path',
        type: 'file',
        filters: [
          { name: 'Model Files', extensions: ['ckpt', 'safetensors', 'pt', 'pth', 'bin'] },
          { name: 'All Files', extensions: ['*'] }
        ],
        placeholder: '/path/to/model.ckpt',
        required: true
      },
      {
        key: 'config_path',
        label: 'Config Path (model_config.json)',
        type: 'file',
        filters: [{ name: 'JSON Files', extensions: ['json'] }],
        placeholder: '/path/to/model_config.json',
        required: true
      }
    ]
  }
]
