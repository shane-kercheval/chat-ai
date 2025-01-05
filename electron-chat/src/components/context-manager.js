import { UIManager } from '../utils/ui-utils.js';
import { generateUUID } from '../utils/utils.js';
import { AttachmentManager } from './attachment-manager.js';
import { AttachmentStore } from '../stores/attachment-store.js';

export class ContextManager extends AttachmentManager {
    constructor(defaultContexts = [], contextsList) {
        super();
        this.elements.contextsList = contextsList;
        this.attachmentStore = AttachmentStore.getInstance();

        const savedContexts = this.loadContextsFromStorage();
        this.contexts = savedContexts || defaultContexts;

        this.initialize();
    }

    initialize() {
        try {
            this.populateContexts();
            this.setupFormHandlers();
        } catch (error) {
            console.error('Error initializing contexts:', error);
            UIManager.showNotification('Error loading contexts', 'error');
        }
    }

    setupFormHandlers() {
        const addButton = document.getElementById('add-context-button');
        const formDiv = document.getElementById('add-context-form');
        const saveButton = document.getElementById('save-context');
        const cancelButton = document.getElementById('cancel-context');

        addButton.addEventListener('click', () => {
            formDiv.style.display = 'block';
        });
        cancelButton.addEventListener('click', () => this.resetForm());
        saveButton.addEventListener('click', () => {
            this.handleNewContext();
            this.resetForm();
        });
    }

    resetForm() {
        const formDiv = document.getElementById('add-context-form');
        const nameInput = formDiv.querySelector('#context-name');
        const typeSelect = formDiv.querySelector('#context-type');
        formDiv.style.display = 'none';
        nameInput.value = '';
        typeSelect.selectedIndex = 0;
    }

    handleNewContext() {
        const form = document.getElementById('add-context-form');
        const name = document.getElementById('context-name').value.trim();
        const type = document.getElementById('context-type').value;
    
        if (!name) {
            UIManager.showNotification('Context name is required', 'error');
            return;
        }
    
        if (this.contexts.some(context => context.name === name)) {
            UIManager.showNotification('A context with this name already exists', 'error');
            return;
        }
    
        const context = {
            id: generateUUID(),
            name,
            type,
            resources: [],
            customInstructions: ''
        };

        this.contexts.push(context);
        const contextDiv = this.createContextItem(context);
        this.elements.contextsList.appendChild(contextDiv);
        form.style.display = 'none';
        this.resetForm();
        this.saveContextsToStorage();
        UIManager.showNotification('Context added', 'success');
    }

    getSearchResults(query) {
        const matchingContexts = this.contexts.filter(context =>
            (query === '' || context.name.toLowerCase().includes(query)) &&
            !this.isItemAttached(context.id)
        );
        return matchingContexts.map(context => {
            const div = this.createSearchResultItem(context, 'context', 'C');
            div.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.attachContext(context, false);
            };
            return div;
        });
    }

    createContextItem(context) {
        const div = document.createElement('div');
        div.className = 'context-item resource-item';
        div.dataset.id = context.id;

        const header = this.createContextHeader(context);
        const configSection = this.createConfigSection(context);
        
        div.appendChild(header);
        div.appendChild(configSection);
        
        header.addEventListener('click', () => {
            const isExpanded = configSection.style.display === 'block';
            configSection.style.display = isExpanded ? 'none' : 'block';
            div.classList.toggle('expanded', !isExpanded);
        });

        return div;
    }

    createContextHeader(context) {
        const header = document.createElement('div');
        header.className = 'context-header';
        
        const content = document.createElement('div');
        content.className = 'context-content resource-content';
        
        const name = document.createElement('div');
        name.className = 'context-name resource-name';
        name.textContent = context.name;
        name.title = context.name;
        
        const type = document.createElement('div');
        type.className = 'context-type resource-path';
        type.textContent = context.type;
        type.title = context.type;
        
        content.appendChild(name);
        content.appendChild(type);
    
        const useButton = this.createUseButton(context, this.attachContext.bind(this));
        const deleteButton = this.createDeleteButton(context);
    
        header.appendChild(content);
        header.appendChild(useButton);
        header.appendChild(deleteButton);
    
        return header;
    }

    createUseButton(context) {
        const useButton = document.createElement('button');
        useButton.className = 'use-button';
        useButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
        `;
        
        // Create a function reference we can remove if needed
        const handleClick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            // Remove the event listener immediately to prevent multiple rapid clicks
            useButton.removeEventListener('click', handleClick);
            this.attachContext(context);
            // Re-add the event listener after a short delay
            setTimeout(() => {
                useButton.addEventListener('click', handleClick);
            }, 100);
        };

        useButton.addEventListener('click', handleClick);
        return useButton;
    }

    createDeleteButton(context) {
        const deleteButton = document.createElement('button');
        deleteButton.className = 'delete-button';
        deleteButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 96 960 960">
                <path fill="currentColor" d="M300 796q0 26 18 43t43 17q26 0 44-17t18-43V516q0-26-18-43t-44-17q-25 0-43 17t-18 43v280Zm240 0q0 26 18 43t43 17q26 0 43-17t17-43V516q0-26-17-43t-43-17q-26 0-43 17t-18 43v280ZM180 256v-60h600v60h136v80H44v-80h136Zm136 720q-41 0-70.5-29.5T216 876V396h528v480q0 41-29.5 70.5T644 976H316Z"/>
            </svg>
        `;
        deleteButton.onclick = (e) => {
            e.stopPropagation();
            const index = this.contexts.findIndex(c => c.id === context.id);
            if (index !== -1) {
                this.contexts.splice(index, 1);
                this.saveContextsToStorage();
                const contextElement = document.querySelector(`.context-item[data-id="${context.id}"]`);
                if (contextElement) {
                    contextElement.remove();
                }
            }
        };
        return deleteButton;
    }

    createConfigSection(context) {
        const configSection = document.createElement('div');
        configSection.className = 'context-config';
        configSection.style.display = 'none';
    
        const customInstructionsSection = this.createCustomInstructionsSection(context);
        const resourcesSection = this.createResourcesSection(context);

        configSection.appendChild(customInstructionsSection);
        configSection.appendChild(resourcesSection);

        return configSection;
    }

    createCustomInstructionsSection(context) {
        const section = document.createElement('div');
        section.className = 'config-section';
        section.innerHTML = `
            <h3>Custom Instructions:</h3>
            <textarea 
                class="custom-instructions" 
                rows="4" 
                placeholder="Enter any custom instructions for this context..."
            >${context.customInstructions || ''}</textarea>
        `;

        const textarea = section.querySelector('.custom-instructions');
        textarea.addEventListener('change', () => {
            context.customInstructions = textarea.value;
            this.saveContextsToStorage();
        });

        return section;
    }

    createResourcesSection(context) {
        const section = document.createElement('div');
        section.className = 'config-section resources-section';
        section.innerHTML = `
            <h3>Resources</h3>
            <div class="resources-add-container">
                <div class="resource-drop-zone">
                    <span>Drag files to add resources</span>
                </div>
            </div>
            <div class="resources-list"></div>
        `;

        const clearResourcesButton = document.createElement('button');
        clearResourcesButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line>
            </svg>
        `;
        clearResourcesButton.className = 'context-resource-button context-resource-clear';
        clearResourcesButton.onclick = (e) => {
            e.stopPropagation();
            const resourcesList = section.querySelector('.resources-list');
            resourcesList.innerHTML = '';
            context.resources = [];
            this.saveContextsToStorage();
        };

        const addResourceButton = document.createElement('button');
        addResourceButton.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="12" y1="5" x2="12" y2="19"></line>
                <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
        `;
        addResourceButton.className = 'context-resource-button context-resource-add';
        addResourceButton.onclick = (e) => {
            e.stopPropagation();
            this.showAddResourceDialog(context);
        };

        const addContainer = section.querySelector('.resources-add-container');
        addContainer.appendChild(clearResourcesButton);
        addContainer.appendChild(addResourceButton);

        const dropZone = section.querySelector('.resource-drop-zone');
        this.setupResourceDropZone(dropZone, context);

        // Populate existing resources
        const resourcesList = section.querySelector('.resources-list');
        if (context.resources) {
            context.resources.forEach(resource => {
                resourcesList.appendChild(this.createResourceItem(resource, context));
            });
        }

        return section;
    }

    setupResourceDropZone(dropZone, context) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        dropZone.addEventListener('dragover', () => {
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', (e) => {
            if (!dropZone.contains(e.relatedTarget)) {
                dropZone.classList.remove('drag-over');
            }
        });

        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');

            try {
                const { webUtils } = require('electron');
                const files = e.dataTransfer.files;
                const resourcesList = dropZone.closest('.resources-section').querySelector('.resources-list');

                for (const file of Array.from(files)) {
                    const filePath = webUtils.getPathForFile(file);
                    console.log('File path from webUtils:', filePath);

                    if (!filePath) {
                        console.error('Could not get path for file:', file.name);
                        continue;
                    }

                    const fs = require('fs');
                    const stats = fs.statSync(filePath);
                    const resource = {
                        id: String(Date.now()),
                        type: stats.isDirectory() ? 'DIRECTORY' : 'FILE',
                        path: filePath,
                        name: file.name
                    };

                    // Add validation
                    if (file.size > 50 * 1024 * 1024) { // 50 MB limit
                        UIManager.showNotification(
                            'Error: File size exceeds 50 MB limit',
                            'error'
                        );
                        continue;
                    }

                    // Check if resource already exists
                    if (context.resources?.some(r => r.path === resource.path)) {
                        UIManager.showNotification(
                            `Warning: Resource "${resource.name}" is already attached.`,
                            'warning'
                        );
                        continue;
                    }

                    if (!context.resources) context.resources = [];
                    context.resources.push(resource);
                    this.saveContextsToStorage(); 
                    this.emit('resourceAdded', resource);
                    resourcesList.appendChild(this.createResourceItem(resource, context));
                }
            } catch (error) {
                console.error('Error processing resources:', error);
                UIManager.showNotification('Error adding resources', 'error');
            }
        });
    }

    getDomainFromUrl(url) {
        try {
            // Try using URL API first
            const domain = new URL(url).hostname;
            // Remove www. if present
            return domain.replace(/^www\./, '');
        } catch {
            // Fallback to regex if URL is malformed
            const match = url.match(/^(?:https?:\/\/)?(?:www\.)?([^\/]+)/i);
            return match ? match[1] : url;
        }
    }

    createResourceItem(resource, context) {
        const div = document.createElement('div');
        div.className = 'resource-item';
        div.dataset.id = resource.id;
        
        const iconContainer = this.getResourceIcon(resource.type);
        
        const content = document.createElement('div');
        content.className = 'resource-content';
        
        const name = document.createElement('div');
        name.className = 'resource-name';
        
        const path = document.createElement('div');
        path.className = 'resource-path';
        
        if (resource.type === 'WEBPAGE') {
            name.textContent = this.getDomainFromUrl(resource.path);
            path.textContent = resource.path;
        } else {
            name.textContent = resource.name;
            path.textContent = resource.path !== resource.name ? resource.path : '';
        }
        name.title = resource.path;
        path.title = resource.path;
            
        content.appendChild(name);
        content.appendChild(path);
        
        const deleteButton = document.createElement('button');
        deleteButton.className = 'remove-file';
        deleteButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M15 9l-6 6M9 9l6 6"/>
            </svg>
        `;
        deleteButton.addEventListener('click', (e) => {
            e.stopPropagation();
            const index = context.resources.findIndex(r => r.id === resource.id);
            if (index !== -1) {
                context.resources.splice(index, 1);
                div.remove();
            }
        });
        
        div.appendChild(iconContainer);
        div.appendChild(content);
        div.appendChild(deleteButton);
        
        return div;
    }

    getResourceIcon(type) {
        const container = document.createElement('div');
        container.className = 'resource-icon-container';
        container.style.position = 'relative';

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '16');
        svg.setAttribute('height', '16');
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('fill', 'none');
        svg.setAttribute('stroke', 'currentColor');
        svg.setAttribute('stroke-width', '2');

        let path;
        switch (type) {
            case 'FILE':
                path = 'M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z';
                break;
            case 'DIRECTORY':
                path = 'M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z';
                break;
            case 'WEBPAGE':
                path = 'M21 12a9 9 0 11-18 0 9 9 0 0118 0z M3.6 9h16.8 M3.6 15h16.8';
                break;
        }

        const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        pathEl.setAttribute('d', path);
        svg.appendChild(pathEl);

        const tooltip = document.createElement('span');
        tooltip.className = 'icon-tooltip';
        tooltip.textContent = type;

        container.appendChild(svg);
        container.appendChild(tooltip);
        
        return container;
    }

    showAddResourceDialog(context) {
        const dialog = document.createElement('div');
        dialog.className = 'resource-dialog';
        dialog.innerHTML = `
            <div class="dialog-content">
                <h3>Add Resource</h3>
                <div class="form-group">
                    <label for="resource-type">Type:</label>
                    <select id="resource-type">
                        <option value="FILE">File</option>
                        <option value="DIRECTORY">Directory</option>
                        <option value="WEBPAGE">Webpage</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="resource-path">Path:</label>
                    <input type="text" id="resource-path" placeholder="Enter path or URL">
                </div>
                <div class="button-group">
                    <button id="save-resource">Save</button>
                    <button id="cancel-resource">Cancel</button>
                </div>
            </div>
        `;
    
        document.body.appendChild(dialog);
    
        const saveButton = dialog.querySelector('#save-resource');
        const cancelButton = dialog.querySelector('#cancel-resource');
    
        saveButton.onclick = () => {
            const type = dialog.querySelector('#resource-type').value;
            const path = dialog.querySelector('#resource-path').value;
    
            if (!path) {
                alert('Path is required');
                return;
            }

            const name = path.split('/').pop() || path;
            const resource = {
                id: String(Date.now()),
                type,
                path,
                name
            };

            if (!context.resources) context.resources = [];
            context.resources.push(resource);
            this.saveContextsToStorage(); 
            this.emit('resourceAdded', resource);

            const contextItem = document.querySelector(`.context-item[data-id="${context.id}"]`);
            const resourcesList = contextItem.querySelector('.resources-list');
            resourcesList.appendChild(this.createResourceItem(resource, context));

            document.body.removeChild(dialog);
        };
    
        cancelButton.onclick = () => {
            document.body.removeChild(dialog);
        };
    }


    attachContext(context, showSuccess = true) {
        try {
            if (this.isItemAttached(context.id)) {
                UIManager.showNotification('Context already attached', 'info');
                return;
            }
            const div = this.createAttachmentItem(context, 'context', 'C');
            const removeButton = div.querySelector('.remove-file');
            removeButton.onclick = () => this.detachContext(context.id, div);
    
            this.elements.attachedItems.appendChild(div);
            this.attachmentStore.addAttachment('contexts', {
                key: context.id,
                value: context
            });
            if (showSuccess) {
                UIManager.showNotification('Context attached', 'success');
            }

            const searchResults = this.elements.searchResults;
            const searchItem = searchResults.querySelector(`.attachment-item[data-id="${context.id}"]`);
            if (searchItem) searchItem.remove();
        } catch (error) {
            console.error('Error attaching context:', error);
            UIManager.showNotification('Error attaching context', 'error');
        }
    }

    detachContext(contextId, element) {
        try {
            element.remove();
            this.attachmentStore.removeAttachment('contexts', contextId);
            this.refreshSearch();
        } catch (error) {
            console.error('Error detaching context:', error);
            UIManager.showNotification('Error removing context', 'error');
        }
    }

    populateContexts() {
        this.elements.contextsList.innerHTML = '';
        this.contexts.forEach(context => {
            const contextDiv = this.createContextItem(context);
            this.elements.contextsList.appendChild(contextDiv);
        });
    }

    getAttachedContexts() {
        const attachedElements = this.elements.attachedItems.querySelectorAll('.attachment-item[data-type="context"]');
        return Array.from(attachedElements).map(element => {
            const context = this.contexts.find(c => c.id === element.dataset.id);
            return context;
        }).filter(Boolean);
    }

    clearAttachments() {
        this.elements.attachedItems.innerHTML = '';
        this.attachmentStore.clearAttachments('contexts');
    }

    saveContextsToStorage() {
        try {
            localStorage.setItem('contexts', JSON.stringify(this.contexts));
        } catch (error) {
            console.error('Error saving contexts to storage:', error);
            UIManager.showNotification('Error saving contexts', 'error');
        }
    }
    
    loadContextsFromStorage() {
        try {
            const savedContexts = localStorage.getItem('contexts');
            if (savedContexts) {
                return JSON.parse(savedContexts);
            }
            return null;
        } catch (error) {
            console.error('Error loading contexts from storage:', error);
            UIManager.showNotification('Error loading saved contexts', 'error');
            return null;
        }
    }
}

