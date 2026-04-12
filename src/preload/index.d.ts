import { ElectronAPI } from '@electron-toolkit/preload'

interface API {
  getRunMode: () => Promise<'local' | 'remote'>
  getHealth: () => Promise<any>
  getProjects: () => Promise<string[]>
  getModels: () => Promise<string[]>
  openProject: () => Promise<string | null>
  newProject: () => Promise<string | null>
  selectDirectory: () => Promise<string | null>
  getRecentProjects: () => Promise<string[]>
  removeRecentProject: (projectPath: string) => Promise<void>
  addRecentProject: (projectPath: string) => Promise<void>
  loadProject: (projectData: { project_path?: string; project_name?: string }) => Promise<any>
  createProject: (projectData: { project_path?: string; project_name?: string }) => Promise<any>
  getGraphData: (viewMode: 'batch' | 'cluster') => Promise<any>
  get_generate_form_config: (adapterName: string) => Promise<any>
  get_import_form_config: (adapterName: string) => Promise<any>
  getAdapterConfig: (adapterName: string) => Promise<any>
  getAudioFile: (
    filename: string
  ) => Promise<{ buffer: Buffer; mimeType: string } | null>
  openFile: (options?: { title?: string; filters?: { name: string; extensions: string[] }[] }) => Promise<string | null>
  importModel: (data: any) => Promise<any>
  generate: (data: any) => Promise<any>
  removeElement: (elementId: string) => Promise<any>
  logMessage: (message: string) => Promise<any>
  pollJobStatus: (jobId: string) => Promise<any>
  updateEmbeddings: () => Promise<any>
  batchElements: (memberIds: string[]) => Promise<any>
  addExternalSource: (sourcePath: string) => Promise<any>
  expandPath: (pathNodeId: string) => Promise<any>
  cancelJob: (jobId: string) => Promise<any>
  saveNodePositions: (projectName: string, positions: Record<string, { x: number; y: number }>) => Promise<void>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: API
  }
}
