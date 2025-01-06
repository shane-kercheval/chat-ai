import { UIManager } from '../utils/ui-utils.js';
import { EventEmitter } from '../utils/event-emitter.js';
import { AttachmentStore } from '../stores/attachment-store.js';


export class FileDropManager extends EventEmitter {
    constructor() {
        super();
        // this.attachedFiles = new Map();
        this.attachmentStore = AttachmentStore.getInstance();
        this.elements = {
            dropZone: document.getElementById('drop-zone'),
            attachedFiles: document.getElementById('attached-files')
        };
        
        this.initialize();
    }

    initialize() {
        this.setupDropZone();
    }

    setupDropZone() {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.elements.dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
    
        this.elements.dropZone.addEventListener('dragover', () => {
            this.elements.dropZone.classList.add('drag-over');
        });
    
        this.elements.dropZone.addEventListener('dragleave', (e) => {
            if (!this.elements.dropZone.contains(e.relatedTarget)) {
                this.elements.dropZone.classList.remove('drag-over');
            }
        });
    
        this.elements.dropZone.addEventListener('drop', async (e) => {
            this.elements.dropZone.classList.remove('drag-over');
        
            try {
                const { webUtils } = require('electron');
                const files = e.dataTransfer.files;
                const filePromises = [];
        
                for (const file of Array.from(files)) {
                    const filePath = webUtils.getPathForFile(file);
                    console.log('File path from webUtils:', filePath);
                    
                    if (!filePath) {
                        console.error('Could not get path for file:', file.name);
                        continue;
                    }
        
                    // Check if file is already attached
                    const resources = this.attachmentStore.getAttachments('resources');
                    if (resources.some(r => r.path === filePath)) {
                        UIManager.showNotification(
                            `Warning: File "${file.name}" is already attached.`,
                            'warning'
                        );
                        continue;
                    }
        
                    const fs = require('fs');
                    const stats = fs.statSync(filePath);
                    // Check file size
                    if (!stats.isDirectory() && file.size > 50 * 1024 * 1024) { // 50 MB limit
                        UIManager.showNotification(
                            'Error: File size exceeds 50 MB limit',
                            'error'
                        );
                        continue;
                    }
                    filePromises.push(this.processFile(file, filePath, file.name, stats.isDirectory()));
                }
        
                await Promise.all(filePromises);
        
                if (filePromises.length > 0) {
                    UIManager.showNotification(
                        `Successfully attached ${filePromises.length} file(s)`,
                        'success'
                    );
                }
            } catch (error) {
                console.error('Error processing files:', error);
                UIManager.showNotification('Error attaching files', 'error');
            }
        });
    }
    
    async processFile(file, absolutePath, fileName, isDirectory) {
        try {
            const type = isDirectory ? 'DIRECTORY' : 'FILE';
            const resource = {
                path: absolutePath,
                type: type,
                name: fileName
            };
            this.attachmentStore.addAttachment('resources', {
                key: absolutePath,
                value: resource
            });
            await this.addFileAttachment(resource);
    
        } catch (error) {
            UIManager.showNotification(error.message, 'error');
            throw error;
        }
    }

    async addFileAttachment(resource) {
        const fileDiv = document.createElement('div');
        fileDiv.className = 'file-attachment';
        
        const nameSpan = document.createElement('span');
        nameSpan.className = 'file-name';
        nameSpan.textContent = resource.name;
        nameSpan.title = resource.path;
    
        const removeButton = document.createElement('button');
        removeButton.className = 'remove-file';
        removeButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M15 9l-6 6M9 9l6 6"/>
            </svg>
        `;
        
        removeButton.onclick = () => {
            this.removeFile(resource.path, fileDiv);
        };
    
        fileDiv.appendChild(nameSpan);
        fileDiv.appendChild(removeButton);
        this.elements.attachedFiles.appendChild(fileDiv);
    }

    removeFile(path, fileDiv) {
        fileDiv.remove();
        this.attachmentStore.removeAttachment('resources', path);
    }

    clearAttachments() {
        this.elements.attachedFiles.innerHTML = '';
        this.attachmentStore.clearAttachments('resources');
    }
}
