import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import http from 'http'
import Store from 'electron-store'
import { spawn, ChildProcess } from 'child_process'

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

    pythonBackend = spawn(venvPath, [appPyPath], {
      stdio: ['pipe', 'pipe', 'pipe'] // Use pipes to capture output
    });

    let backendReady = false;
    const handleData = (data: Buffer) => {
      const message = data.toString();
      if (!backendReady && message.includes('Running on http://127.0.0.1:5000')) {
        backendReady = true;
        console.log('Python backend started');
        resolve();
      }
    }

    pythonBackend.stdout?.on('data', (data) => {
      console.log(`Python Backend: ${data}`);
      handleData(data)
    });

    pythonBackend.stderr?.on('data', (data) => {
      console.error(`Python Backend Error: ${data}`);
      handleData(data)
    });

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
      sandbox: false
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

  ipcMain.handle('getRecentProjects', async () => {
    return (store.get('recentProjects', []) as string[])
  })

  ipcMain.handle('addRecentProject', async (_event, projectPath) => {
    const recentProjects = (store.get('recentProjects', []) as string[]).filter(p => p !== projectPath)
    recentProjects.unshift(projectPath)
    store.set('recentProjects', recentProjects.slice(0, 10))
  })

  ipcMain.handle('loadProjectAndGetData', async (_event, projectPath) => {
    // 1. Load the project
    const postData = JSON.stringify({ project_path: projectPath })
    const postOptions = {
      hostname: '127.0.0.1',
      port: 5000,
      path: '/load',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
      }
    }

    const postRequest = new Promise<void>((resolve, reject) => {
      const req = http.request(postOptions, (res) => {
        if (res.statusCode === 200) {
          resolve()
        } else {
          res.on('data', d => console.error(d.toString()))
          reject(new Error(`Failed to load project. Status code: ${res.statusCode}`))
        }
      })
      req.on('error', (e) => reject(e))
      req.write(postData)
      req.end()
    })

    await postRequest

    // 2. Get the graph data
    const getOptions = {
      hostname: '127.0.0.1',
      port: 5000,
      path: '/graph',
      method: 'GET'
    }

    const getRequest = new Promise((resolve, reject) => {
      const req = http.request(getOptions, (res) => {
        let data = ''
        res.on('data', (chunk) => {
          data += chunk
        })
        res.on('end', () => {
          if (res.statusCode === 200) {
            try {
              resolve(JSON.parse(data))
            } catch (e) {
              reject(new Error('Failed to parse graph data from backend.'))
            }
          } else {
            reject(new Error(`Failed to get graph data. Status code: ${res.statusCode}`))
          }
        })
      })
      req.on('error', (e) => reject(e))
      req.end()
    })

    return await getRequest
  })


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