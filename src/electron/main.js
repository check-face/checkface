// Modules to control application life and create native browser window
const {app, BrowserWindow} = require('electron')
const electron = require('electron')
const path = require('path')
const sha256File = require('sha256-file');

if (process.env.NODE_ENV === 'development') {
  require('electron-reload')(__dirname, {
    electron: require(`${__dirname}/node_modules/electron`)
  });
}
// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow

function createWindow () {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    backgroundColor: '#ECECEC',
    useContentSize: true,
    width: 565,
    height: 500,
    webPreferences: {
      nodeIntegration: true
    }
  })
  let args = process.argv;
  if (process.env.NODE_ENV === 'development') {
    args = args.slice(1);
  }
  else {
    mainWindow.setMenu(null);
  }
  if(process.argv.length >= 2) {
    global.hashdata = { filename: process.argv[process.argv.length - 1] };
    sha256File(global.hashdata.filename, function (error, sum) {
      if (error) {
        console.log("Error getting sum: ", error);
        global.hashdata.message = "Error calculating sha256 for file.";
      }
      else {
        global.hashdata.message = sum;
        global.hashdata.sum = sum;
      }
      console.log("SHA256:", sum);
      mainWindow.loadFile('index.html')
    })
    console.log("Filename:", global.hashdata.filename);
  }
  else {
    console.log("Args", process.argv);
    global.hashdata = { message: "Please select a file to hash as a command line argument." };
    mainWindow.loadFile('index.html')
  }
  
  // and load the index.html of the app.



  // Open the DevTools.
  // mainWindow.webContents.openDevTools()

  // Emitted when the window is closed.
  mainWindow.on('closed', function () {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', function () {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) createWindow()
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
