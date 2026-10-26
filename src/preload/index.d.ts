import { ElectronAPI } from '@electron-toolkit/preload'
import type {
  EngineConflict,
  EngineSettings,
  EngineState,
  ForkResult,
  ProjectOrigin
} from '../renderer/src/utils/types'

interface API {
  getRunMode: () => Promise<'local' | 'remote'>
  getHealth: () => Promise<any>
  getProjects: () => Promise<string[]>
  getModels: () => Promise<string[]>
  openProject: () => Promise<string | null>
  newProject: (defaultPath?: string) => Promise<string | null>
  getDefaultProjectLocation: () => Promise<string>
  selectDirectory: () => Promise<string | null>
  selectSavePath: (defaultName?: string) => Promise<string | null>
  getRecentProjects: () => Promise<string[]>
  removeRecentProject: (projectPath: string) => Promise<void>
  addRecentProject: (projectPath: string) => Promise<void>
  loadProject: (projectData: {
    project_path?: string
    project_name?: string
  }) => Promise<{ project_name: string; project_path: string; origin?: ProjectOrigin | null }>
  createProject: (projectData: { project_path?: string; project_name?: string }) => Promise<any>
  forkProject: (elementIds: string[]) => Promise<ForkResult>
  createRoom: (elementIds: string[]) => Promise<{ room_id: string }>
  dissolveRoom: (roomId: string) => Promise<{ member_ids: string[] }>
  setViewRoom: (roomId: string | null) => Promise<{ room_id: string | null }>
  // check probes the remote engine service and fills in EngineState.connection
  getEngine: (check?: boolean) => Promise<EngineState>
  setEngine: (settings: EngineSettings, force?: boolean) => Promise<EngineState | EngineConflict>
  // Reapplies the saved engine choice; null when none has been saved
  restoreEngine: () => Promise<EngineState | EngineConflict | null>
  getGraphData: () => Promise<any>
  get_generate_form_config: (adapterName: string) => Promise<any>
  get_import_form_config: (adapterName: string) => Promise<any>
  getAdapterConfig: (adapterName: string) => Promise<any>
  getAudioFile: (
    filename: string
  ) => Promise<{ buffer: Buffer; mimeType: string } | null>
  getImageFile: (
    filename: string
  ) => Promise<{ buffer: Buffer; mimeType: string } | null>
  registerGrating: (data: { name: string; checkpoint_path: string; base_model_id: string }) => Promise<any>
  getModelLayers: (modelId: string) => Promise<any>
  createGrating: (data: any) => Promise<any>
  startEvolution: (data: unknown) => Promise<unknown>
  expressIndividual: (data: { individual_id: string } | unknown) => Promise<unknown>
  openFile: (options?: { title?: string; filters?: { name: string; extensions: string[] }[] }) => Promise<string | null>
  getSharedModels: () => Promise<any>
  importModel: (data: any) => Promise<any>
  exportAudio: (names: string[], exportDir?: string) => Promise<any>
  getOperations: () => Promise<any>
  executeOperation: (data: any) => Promise<any>
  getExemplarPresets: (individualIds: string[]) => Promise<any>
  removeElement: (elementId: string, keepChildren?: boolean) => Promise<any>
  removeElements: (elementIds: string[], keepChildren?: boolean) => Promise<any>
  updateLabels: () => Promise<any>
  repairEdges: () => Promise<any>
  pollJobStatus: (jobId: string) => Promise<any>
  updateEmbeddings: () => Promise<any>
  groupElements: (memberIds: string[]) => Promise<any>
  updateGroup: (groupId: string, memberIds: string[]) => Promise<any>
  addExternalSource: (sourcePath: string) => Promise<any>
  rescanSource: (sourceName: string) => Promise<any>
  expandPath: (pathNodeId: string) => Promise<any>
  cancelJob: (jobId: string) => Promise<any>
  saveNodePositions: (projectName: string, positions: Record<string, { x: number; y: number }>) => Promise<void>
  updateElement: (elementName: string, attributes: Record<string, any>) => Promise<any>
  exportSharedModel: (modelId: string) => Promise<unknown>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: API
  }
}
