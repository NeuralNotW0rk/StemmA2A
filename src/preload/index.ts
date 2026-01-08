import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  openProject: (): Promise<string | null> => ipcRenderer.invoke('dialog:openProject'),
  getRecentProjects: (): Promise<string[]> => ipcRenderer.invoke('getRecentProjects'),
  addRecentProject: (projectPath: string): Promise<void> =>
    ipcRenderer.invoke('addRecentProject', projectPath),
  loadProjectAndGetData: (projectPath: string): Promise<any> =>
    ipcRenderer.invoke('loadProjectAndGetData', projectPath)
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
