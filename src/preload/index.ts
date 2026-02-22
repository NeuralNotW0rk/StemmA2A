import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  openProject: (): Promise<string | null> => ipcRenderer.invoke('dialog:openProject'),
  newProject: (): Promise<string | null> => ipcRenderer.invoke('dialog:newProject'),
  getRecentProjects: (): Promise<string[]> => ipcRenderer.invoke('getRecentProjects'),
  removeRecentProject: (projectPath: string): Promise<void> =>
    ipcRenderer.invoke('removeRecentProject', projectPath),
  addRecentProject: (projectPath: string): Promise<void> =>
    ipcRenderer.invoke('addRecentProject', projectPath),
  loadProject: (projectPath: string): Promise<any> =>
    ipcRenderer.invoke('loadProject', projectPath),
  createProject: (projectPath: string): Promise<any> =>
    ipcRenderer.invoke('createProject', projectPath),
  getGraphData: (viewMode: 'batch' | 'cluster'): Promise<any> =>
    ipcRenderer.invoke('getGraphData', viewMode),
  get_generate_form_config: (engineName: string): Promise<any> =>
    ipcRenderer.invoke('get_generate_form_config', engineName),
  get_import_form_config: (engineName: string): Promise<any> =>
    ipcRenderer.invoke('get_import_form_config', engineName),
  getEngineConfig: (engineName: string): Promise<any> =>
    ipcRenderer.invoke('getEngineConfig', engineName),
  getAudioFile: (filename: string): Promise<string | null> =>
    ipcRenderer.invoke('getAudioFile', filename),
  openFile: (options: unknown) => ipcRenderer.invoke('dialog:openFile', options),
  importModel: (data: unknown) => ipcRenderer.invoke('importModel', data),
  generate: (data: unknown) => ipcRenderer.invoke('generate', data),
  logMessage: (message: string): Promise<any> => ipcRenderer.invoke('logMessage', message)
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.api = api
}
