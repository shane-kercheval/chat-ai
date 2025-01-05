const { app, BrowserWindow } = require('electron');
const { ipcMain } = require('electron');
const path = require('path');
const { setupIpcHandlers } = require('./ipcHandlers');

function createWindow() {
    const mainWindow = new BrowserWindow({
        title: 'Chat App',
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            webSecurity: true
        },
    });

    mainWindow.loadFile(path.join(__dirname, '../index.html'));
    setupIpcHandlers(mainWindow);

    // Open the DevTools by default
    mainWindow.webContents.openDevTools();
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Add this with your other IPC handlers
ipcMain.handle('handle-file-drop', async (event, filePaths) => {
    return filePaths.map(filePath => ({
        path: filePath,
        name: path.basename(filePath)
    }));
});
