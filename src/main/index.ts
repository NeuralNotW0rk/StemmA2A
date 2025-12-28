// src/main/index.ts
import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { spawn, ChildProcess } from 'child_process'

let mainWindow: BrowserWindow | null = null
let pythonBackend: ChildProcess | null = null

// Backend management
function startPythonBackend(): Promise<void> {
  return new Promise((resolve, reject) => {
    // Start the Flask backend server
    pythonBackend = spawn('python', ['StemmA2A/backend/app.py'], {
      cwd: join(__dirname, '../../..'),
      stdio: ['inherit', 'inherit', 'inherit']
    })

    pythonBackend.on('error', (error) => {
      console.error('Failed to start Python backend:', error)
      reject(error)
    })

    // Give the server time to start
    setTimeout(() => {
      console.log('Python backend started')
      resolve()
    }, 3000)
  })
}

function stopPythonBackend(): void {
  if (pythonBackend) {
    pythonBackend.kill()
    pythonBackend = null
  }
}

function createWindow(): void {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    show: false,
    autoHideMenuBar: true,
    titleBarStyle: 'hiddenInset',
    vibrancy: 'under-window',
    visualEffectState: 'active',
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true,
      enableRemoteModule: false,
      webSecurity: false // Allow local file access for audio files
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // Load the app
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  // Open DevTools in development
  if (is.dev) {
    mainWindow.webContents.openDevTools()
  }
}

// IPC handlers for communication with renderer
ipcMain.handle('backend-request', async (event, { endpoint, method = 'GET', data = null }) => {
  const url = `http://localhost:5000${endpoint}`
  const options: RequestInit = { method }

  if (data && method !== 'GET') {
    options.headers = { 'Content-Type': 'application/json' }
    options.body = JSON.stringify(data)
  }

  try {
    const response = await fetch(url, options)
    return await response.json()
  } catch (error) {
    console.error('Backend request failed:', error)
    throw error
  }
})

ipcMain.handle('load-project', async (event, projectName: string) => {
  try {
    const formData = new FormData()
    formData.append('project_name', projectName)
    
    const response = await fetch('http://localhost:5000/load', {
      method: 'POST',
      body: formData
    })
    
    return await response.json()
  } catch (error) {
    console.error('Failed to load project:', error)
    throw error
  }
})

ipcMain.handle('get-audio-file', async (event, filename: string) => {
  // Return audio file path for local access
  const audioPath = join(app.getPath('userData'), 'projects', filename)
  return audioPath
})

// App lifecycle
app.whenReady().then(async () => {
  electronApp.setAppUserModelId('com.stemma2a.app')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // Start backend before creating window
  try {
    await startPythonBackend()
    createWindow()
  } catch (error) {
    console.error('Failed to start application:', error)
    app.quit()
  }

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  stopPythonBackend()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopPythonBackend()
})