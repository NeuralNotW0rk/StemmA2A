import { ElectronAPI } from '@electron-toolkit/preload'

interface API {
  openProject: () => Promise<string | null>
  newProject: () => Promise<string | null>
  getRecentProjects: () => Promise<string[]>
  addRecentProject: (projectPath: string) => Promise<void>
  loadProject: (projectPath: string) => Promise<any>
  createProject: (projectPath: string) => Promise<any>
  getGraphData: (viewMode: 'batch' | 'cluster') => Promise<any>
  getAudioFile: (
    filename: string
  ) => Promise<{ buffer: Buffer; mimeType: string } | null>
  openFile: (options?: { title?: string; filters?: { name: string; extensions: string[] }[] }) => Promise<string | null>
  importModel: (data: any) => Promise<any>
  logMessage: (message: string) => Promise<any>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: API
  }
}
