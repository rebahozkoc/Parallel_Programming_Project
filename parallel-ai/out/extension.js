"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require("vscode");
// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
function activate(context) {
    // Use the console to output diagnostic information (console.log) and errors (console.error)
    // This line of code will only be executed once when your extension is activated
    console.log('Congratulations, your extension "parallel-ai" is now active!');
    // The command has been defined in the package.json file
    // Now provide the implementation of the command with registerCommand
    // The commandId parameter must match the command field in package.json
    let currenTime = new Date().toLocaleTimeString();
    let disposable = vscode.commands.registerCommand('parallel-ai.parallel-ai', async () => {
        // The code you place here will be executed every time your command is executed
        // Display a message box to the user
        let editor = vscode.window.activeTextEditor;
        if (editor) {
            let document = editor.document;
            // Get the selection, or set it to the start of the document if there is no selection
            let selection = editor.selection || new vscode.Selection(new vscode.Position(0, 0), new vscode.Position(0, 0));
            // Get the text of the selection
            let text = document.getText(selection);
            // Create a new WebviewPanel
            const panel = vscode.window.createWebviewPanel('textViewer', // Identifies the type of the webview. Used internally
            'Selected Text', // Title of the panel displayed to the user
            vscode.ViewColumn.One, // Editor column to show the new webview panel in.
            {} // Webview options. More on these later.
            );
            // Set the webview's HTML content
            panel.webview.html = `<html><body><p>${text}</p></body></html>`;
        }
        //vscode.window.showInformationMessage('Hello World from parallel_ai by rebahozkoc and elifyildirir!', currenTime);
    });
    context.subscriptions.push(disposable);
}
exports.activate = activate;
// This method is called when your extension is deactivated
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map