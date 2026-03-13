import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  getHealth: (): Promise<any> => ipcRenderer.invoke('getHealth'),
  openProject: (): Promise<string | null> => ipcRenderer.invoke('dialog:openProject'),
  newProject: (): Promise<string | null> => ipcRenderer.invoke('dialog:newProject'),
  getRecentProjects: (): Promise<string[]> => ipcRenderer.invoke('getRecentProjects'),
  removeRecentProject: (projectPath: string): Promise<void> =>
    ipcRenderer.invoke('removeRecentProject', projectPath),
  addRecentProject: (projectPath: string): Promise<void> =>
    ipcRenderer.invoke('addRecentProject', projectPath),
  loadProject: (projectData: any): Promise<any> => ipcRenderer.invoke('loadProject', projectData),
  createProject: (projectData: any): Promise<any> =>
    ipcRenderer.invoke('createProject', projectData),
  getGraphData: (viewMode: 'batch' | 'cluster'): Promise<any> =>
    ipcRenderer.invoke('getGraphData', viewMode),
  get_generate_form_config: (adapterName: string): Promise<any> =>
    ipcRenderer.invoke('get_generate_form_config', adapterName),
  get_import_form_config: (adapterName: string): Promise<any> =>
    ipcRenderer.invoke('get_import_form_config', adapterName),
  getAdapterConfig: (adapterName: string): Promise<any> =>
    ipcRenderer.invoke('getAdapterConfig', adapterName),
  getAudioFile: (filename: string): Promise<string | null> =>
    ipcRenderer.invoke('getAudioFile', filename),
  openFile: (options: unknown) => ipcRenderer.invoke('dialog:openFile', options),
  importModel: (data: unknown) => ipcRenderer.invoke('importModel', data),
  generate: (data: unknown) => ipcRenderer.invoke('generate', data),
  removeElement: (elementId: string): Promise<any> =>
    ipcRenderer.invoke('removeElement', elementId),
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
