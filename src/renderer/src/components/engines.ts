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
  fields: EngineField[]
}

export const engines: EngineConfig[] = [
  {
    id: 'stable-audio-tools',
    name: 'Stable Audio Tools',
    fields: [
      {
        key: 'checkpointPath',
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
        key: 'configPath',
        label: 'Config Path (model_config.json)',
        type: 'file',
        filters: [{ name: 'JSON Files', extensions: ['json'] }],
        placeholder: '/path/to/model_config.json',
        required: true
      }
    ]
  }
]
