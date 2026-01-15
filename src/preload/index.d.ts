import { ElectronAPI } from '@electron-toolkit/preload'

interface API {
  openProject: () => Promise<string | null>
  getRecentProjects: () => Promise<string[]>
  addRecentProject: (projectPath: string) => Promise<void>
  loadProjectAndGetData: (projectPath: string) => Promise<any>
  logMessage: (message: string) => Promise<any>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: API
  }
}
