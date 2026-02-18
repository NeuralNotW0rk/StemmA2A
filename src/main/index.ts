import { app, shell, BrowserWindow, ipcMain, dialog, protocol } from 'electron'
import { join, extname } from 'path'
import { promises as fs } from 'fs'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import Store from 'electron-store'
import { spawn, ChildProcess } from 'child_process'
import readline from 'readline'

let pythonBackend: ChildProcess | null = null
const store = new Store()

// Backend management
function startPythonBackend(): Promise<void> {
  return new Promise((resolve, reject) => {
    const pythonExecutable = process.platform === 'win32' ? 'python.exe' : 'python';
    const venvPath = is.dev
      ? join(app.getAppPath(), 'backend', '.venv', 'Scripts', pythonExecutable)
      : join(process.resourcesPath, 'backend', '.venv', 'Scripts', pythonExecutable);

    const appPyPath = is.dev
      ? join(app.getAppPath(), 'backend', 'app.py')
      : join(process.resourcesPath, 'backend', 'app.py');

    if (is.dev && process.platform === 'win32') {
      // In dev mode on Windows, open a new terminal that stays open
      console.log('Starting Python backend in new window');
      const command = ['/c', 'start', 'cmd.exe', '/k', venvPath, '-u', appPyPath];
      pythonBackend = spawn('cmd.exe', command, {
        cwd: join(app.getAppPath(), 'backend')
      });
      // Resolve after a delay to allow the backend to initialize
      setTimeout(() => {
        console.log('Assuming Python backend is ready');
        resolve();
      }, 5000); // 5 seconds delay
    } else {
      // Original behavior for production or other platforms
      pythonBackend = spawn(venvPath, ['-u', appPyPath], {
        stdio: ['pipe', 'pipe', 'pipe'] // Use pipes to capture output
      });

      let backendReady = false;
      const handleMessage = (message: string) => {
        if (!backendReady && message.includes('Running on http://127.0.0.1:5000')) {
          backendReady = true;
          console.log('Python backend started');
          resolve();
        }
      }

      const stdoutReader = readline.createInterface({ input: pythonBackend.stdout! });
      stdoutReader.on('line', (line) => {
        console.log(`Python Backend: ${line}`);
        handleMessage(line);
      });

      const stderrReader = readline.createInterface({ input: pythonBackend.stderr! });
      stderrReader.on('line', (line) => {
        // Log stderr as regular output, since Flask/Werkzeug logs INFO here
        console.log(`Python Backend: ${line}`);
        handleMessage(line);
      });
    }

    pythonBackend.on('error', (error) => {
      console.error('Failed to start Python backend:', error);
      reject(error);
    });
    
    pythonBackend.on('exit', (code) => {
      console.log(`Python backend exited with code ${code}`);
    });
  });
}

function stopPythonBackend(): void {
  if (pythonBackend) {
    console.log('Stopping Python backend...')
    pythonBackend.kill()
    pythonBackend = null
  }
}

function createWindow(): void {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      webSecurity: true
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(async () => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  protocol.registerFileProtocol('audiop', (request, callback) => {
    const url = request.url.substr(9) // strip 'audiop://'
    callback({ path: decodeURIComponent(url) })
  })

  ipcMain.handle('dialog:newProject', async () => {
    const { canceled, filePath } = await dialog.showSaveDialog({
      title: 'Create New Project',
      defaultPath: 'New Project',
      buttonLabel: 'Create',
      properties: ['createDirectory', 'showOverwriteConfirmation']
    })
    if (canceled) {
      return null
    } else {
      return filePath
    }
  })

  ipcMain.handle('dialog:openProject', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
      properties: ['openDirectory']
    })
    if (canceled) {
      return null
    } else {
      return filePaths[0]
    }
  })

  ipcMain.handle('dialog:openFile', async (_event, options) => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
      title: options?.title || 'Open File',
      properties: ['openFile'],
      filters: options?.filters || []
    })
    if (canceled) {
      return null
    }
    return filePaths[0]
  })

  ipcMain.handle('importModel', async (_event, modelData) => {
    const response = await fetch('http://127.0.0.1:5000/register_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(modelData)
    })
    if (!response.ok) {
      throw new Error(`Failed to import model. Status: ${response.status}`)
    }
    return await response.json()
  })

  ipcMain.handle('getRecentProjects', async () => {
    return (store.get('recentProjects', []) as string[])
  })

  ipcMain.handle('addRecentProject', async (_event, projectPath) => {
    const recentProjects = (store.get('recentProjects', []) as string[]).filter(p => p !== projectPath)
    recentProjects.unshift(projectPath)
    store.set('recentProjects', recentProjects.slice(0, 10))
  })

  ipcMain.handle('getAudioFile', async (_event, filename) => {
    try {
      // 1. Get path from Python backend
      const pathResponse = await fetch(`http://127.00.1:5000/audio_path/${filename}`)
      if (!pathResponse.ok) {
        const errorBody = await pathResponse.text()
        throw new Error(
          `Failed to get audio path from backend. Status: ${pathResponse.status}. Body: ${errorBody}`
        )
      }
      const { path: audioPath } = await pathResponse.json()

      // 2. Read the file into a buffer
      const audioBuffer = await fs.readFile(audioPath)

      // 3. Determine MIME type
      const extension = extname(audioPath).toLowerCase()
      let mimeType = 'audio/wav' // Default
      if (extension === '.mp3') {
        mimeType = 'audio/mpeg'
      } else if (extension === '.ogg') {
        mimeType = 'audio/ogg'
      }

      // 4. Return buffer and mime type
      return { buffer: audioBuffer, mimeType: mimeType }
    } catch (error) {
      console.error('Failed to get audio file:', error)
      return null
    }
  })



  ipcMain.handle('loadProject', async (_event, projectPath) => {
    // Load project
    const loadResponse = await fetch('http://127.0.0.1:5000/load_project', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ project_path: projectPath }),
    });

    // Log status
    console.log(`Main Process: /load_project status: ${loadResponse.status}`);
    if (!loadResponse.ok) {
        const errorText = await loadResponse.text();
        console.error(`Main Process: /load_project error data: ${errorText}`)
        throw new Error(`Failed to load project. Status code: ${loadResponse.status}`);
    }
  })

  ipcMain.handle('createProject', async (_event, projectPath) => {
    // Create the directory if it doesn't exist
    await fs.mkdir(projectPath, { recursive: true })

    // Create project
    const createResponse = await fetch('http://127.0.0.1:5000/create_project', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({ project_path: projectPath }),
    });

    // Log status
    console.log(`Main Process: /create_project status: ${createResponse.status}`);
    if (!createResponse.ok) {
        const errorText = await createResponse.text();
        console.error(`Main Process: /create_project error data: ${errorText}`)
        throw new Error(`Failed to create project. Status code: ${createResponse.status}`);
    }
  })


  ipcMain.handle('getGraphData', async (_event, viewMode) => {
    const endpoint = viewMode === 'cluster' ? '/graph_tsne' : '/graph';
    console.log(`Main Process: getting ${endpoint}`);
    const graphResponse = await fetch(`http://127.0.0.1:5000${endpoint}`);
    console.log(`Main Process: ${endpoint} status: ${graphResponse.status}`);
    if (!graphResponse.ok) {
        const errorText = await graphResponse.text();
        throw new Error(`Failed to get graph data from ${endpoint}. Status code: ${graphResponse.status}. Body: ${errorText}`);
    }

    const graphDataText = await graphResponse.text();
    console.log(`Main Process: ${endpoint} data received, parsing...`);
    try {
        const parsedData = JSON.parse(graphDataText);
        console.log(`Main Process: ${endpoint} data parsed successfully.`);
        return parsedData.graph_data;
    } catch (e) {
        console.error(`Main Process: ${endpoint} JSON parse error:`, e);
        console.error('Main Process: Raw data was:', graphDataText);
        throw new Error('Failed to parse graph data from backend.');
    }
  });


  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })
  
  try {
    await startPythonBackend()
    createWindow()
  } catch (error) {
    console.error('Failed to start application:', error)
    dialog.showErrorBox('Startup Error', 'Failed to start the Python backend. The application will close.')
    app.quit()
  }

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopPythonBackend()
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.