import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  getHealth: (): Promise<any> => ipcRenderer.invoke('getHealth'),
  openProject: (): Promise<string | null> => ipcRenderer.invoke('dialog:openProject'),
  newProject: (): Promise<string | null> => ipcRenderer.invoke('dialog:newProject'),
  selectDirectory: (): Promise<string | null> => ipcRenderer.invoke('dialog:selectDirectory'),
  selectSavePath: (defaultName?: string): Promise<string | null> =>
    ipcRenderer.invoke('dialog:selectSavePath', defaultName),
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
  getAudioFile: (audio_id: string): Promise<string | null> =>
    ipcRenderer.invoke('getAudioFile', audio_id),
  openFile: (options: unknown) => ipcRenderer.invoke('dialog:openFile', options),
  getSharedModels: () => ipcRenderer.invoke('getSharedModels'),
  importModel: (data: unknown) => ipcRenderer.invoke('importModel', data),
  exportAudio: (names: string[], exportDir?: string): Promise<any> =>
    ipcRenderer.invoke('exportAudio', names, exportDir),
  registerGrating: (data: unknown) => ipcRenderer.invoke('registerGrating', data),
  getOperations: (): Promise<any> => ipcRenderer.invoke('getOperations'),
  executeOperation: (data: unknown): Promise<any> => ipcRenderer.invoke('executeOperation', data),
  removeElement: (elementId: string, keepChildren?: boolean): Promise<any> =>
    ipcRenderer.invoke('removeElement', elementId, keepChildren),
  removeElements: (elementIds: string[], keepChildren?: boolean): Promise<any> =>
    ipcRenderer.invoke('removeElements', elementIds, keepChildren),
  logMessage: (message: string): Promise<any> => ipcRenderer.invoke('logMessage', message),
  updateLabels: (): Promise<any> => ipcRenderer.invoke('updateLabels'),
  repairEdges: (): Promise<any> => ipcRenderer.invoke('repairEdges'),
  updateEmbeddings: (): Promise<any> => ipcRenderer.invoke('updateEmbeddings'),
  pollJobStatus: (jobId: string): Promise<any> => ipcRenderer.invoke('pollJobStatus', jobId),
  batchElements: (memberIds: string[]): Promise<any> => ipcRenderer.invoke('batchElements', memberIds),
  updateBatch: (batchId: string, memberIds: string[]): Promise<any> =>
    ipcRenderer.invoke('updateBatch', batchId, memberIds),
  addExternalSource: (sourcePath: string): Promise<any> => ipcRenderer.invoke('addExternalSource', sourcePath),
  expandPath: (pathNodeId: string): Promise<any> => ipcRenderer.invoke('expandPath', pathNodeId),
  cancelJob: (jobId: string): Promise<any> => ipcRenderer.invoke('cancel-job', jobId),
  saveNodePositions: (projectName: string, positions: Record<string, { x: number; y: number }>): Promise<void> =>
    ipcRenderer.invoke('saveNodePositions', projectName, positions),
  updateElement: (elementName: string, attributes: Record<string, any>): Promise<any> =>
    ipcRenderer.invoke('updateElement', elementName, attributes)
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
