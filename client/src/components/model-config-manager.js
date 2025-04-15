import { UIManager } from '../utils/ui-utils.js';
import { AttachmentManager } from './attachment-manager.js';

export class ModelConfigManager extends AttachmentManager {
    constructor(modelConfigs = [], configsList, ipcService) {
        super();
        this.modelConfigs = modelConfigs;
        this.elements.configsList = configsList;
        this.ipcService = ipcService;
        this.supportedModels = [];  // Will be populated in initialize()
        this.initialize();
    }

    async initialize() {
        try {
            // Get supported models first
            const response = await this.ipcService.getSupportedModels();
            this.supportedModels = response.modelsList;
    
            // Update select options in form
            this.updateModelSelectOptions();
            this.populateModelConfigs();
            this.setupFormHandlers();
            // Attach first config by default if available
            if (this.modelConfigs.length > 0) {
                this.attachModelConfig(this.modelConfigs[0], false); // false to not show success message
            }
        } catch (error) {
            console.error('Error initializing model configs:', error);
            UIManager.showNotification('Error loading model configurations', 'error');
        }
    }

    updateModelSelectOptions() {
        const modelSelect = document.getElementById('model-select');
        if (!modelSelect) return;
    
        modelSelect.innerHTML = this.supportedModels
            .map(model => `<option value="${model.name}">${model.displayName}</option>`)
            .join('');
    }

    setupFormHandlers() {
        const addButton = document.getElementById('add-model-config-button');
        const formDiv = document.getElementById('add-model-config-form');
        const saveButton = document.getElementById('save-model-config');
        const cancelButton = document.getElementById('cancel-model-config');
        const modelSelect = formDiv.querySelector('#model-select');
        const serverUrlContainer = formDiv.querySelector('#server-url-container');
        const temperatureSlider = formDiv.querySelector('#temperature-slider');
        const temperatureValue = formDiv.querySelector('#temperature-value');
    
        addButton.addEventListener('click', () => {
            formDiv.style.display = 'block';
        });
    
        modelSelect.addEventListener('change', () => {
            serverUrlContainer.style.display = 
                modelSelect.value === 'openai-compatible-server' ? 'block' : 'none';
        });
    
        const resetForm = () => {
            formDiv.style.display = 'none';
            // Reset text/url inputs
            formDiv.querySelectorAll('input[type="text"], input[type="url"]').forEach(input => {
                input.value = '';
            });
            // Reset select
            modelSelect.selectedIndex = 0;
            // Hide server URL container
            serverUrlContainer.style.display = 'none';
            // Reset temperature slider and its label
            temperatureSlider.value = temperatureSlider.defaultValue;
            temperatureValue.textContent = temperatureSlider.defaultValue;
        };
    
        cancelButton.addEventListener('click', resetForm);
        saveButton.addEventListener('click', () => {
            this.handleNewModelConfig();
            resetForm();
        });
    }

    async handleNewModelConfig() {
        const form = document.getElementById('add-model-config-form');
        const name = document.getElementById('config-name').value.trim();
        const modelName = document.getElementById('model-select').value;
        const temperature = parseFloat(document.getElementById('temperature-slider').value);
        const serverUrl = modelName === 'openai-compatible-server' ? 
            document.getElementById('server-url').value.trim() : null;
    
        if (!name) {
            UIManager.showNotification('Configuration name is required', 'error');
            return;
        }
    
        if (this.modelConfigs.some(config => config.name === name)) {
            UIManager.showNotification('A configuration with this name already exists', 'error');
            return;
        }
    
        try {
            const savedConfig = await this.ipcService.saveModelConfig({
                name,
                modelName,
                temperature,
                serverUrl
            });
            
            const config = {
                id: savedConfig.configId,
                name: savedConfig.configName,
                modelName: savedConfig.config.modelName,
                temperature: savedConfig.config.modelParameters?.temperature,
                serverUrl: savedConfig.config.modelParameters?.serverUrl
            };
    
            this.modelConfigs.push(config);
            const configDiv = this.createModelConfigItem(config);
            this.elements.configsList.appendChild(configDiv);
            
            form.style.display = 'none';
            UIManager.showNotification('Model configuration added', 'success');
        } catch (error) {
            console.error('Error saving model config:', error);
            UIManager.showNotification('Failed to save model configuration', 'error');
        }
    }

    getSearchResults(query) {
        const matchingConfigs = this.modelConfigs.filter(config =>
            (query === '' || config.name.toLowerCase().includes(query)) &&
            !this.isItemAttached(config.id)
        );
        return matchingConfigs.map(config => {
            const div = this.createSearchResultItem(config, 'model-config', 'M');
            div.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.attachModelConfig(config, false);
            };
            return div;
        });
    }

    createModelConfigItem(config) {
        const div = document.createElement('div');
        div.className = 'model-config-item resource-item';
        div.dataset.id = config.id;

        const header = this.createModelConfigHeader(config);
        const configSection = this.createConfigSection(config);
        
        div.appendChild(header);
        div.appendChild(configSection);
        
        header.addEventListener('click', () => {
            const isExpanded = configSection.style.display === 'block';
            configSection.style.display = isExpanded ? 'none' : 'block';
            div.classList.toggle('expanded', !isExpanded);
        });

        return div;
    }

    createModelConfigHeader(config) {
        const header = document.createElement('div');
        header.className = 'model-config-header';
        
        const content = document.createElement('div');
        content.className = 'model-config-content resource-content';
        
        const name = document.createElement('div');
        name.className = 'model-config-name resource-name';
        name.textContent = config.name;
        name.title = config.name;
        
        const modelName = document.createElement('div');
        modelName.className = 'model-config-model resource-path';
        modelName.textContent = config.modelName;
        modelName.title = config.modelName;
        
        content.appendChild(name);
        content.appendChild(modelName);

        const useButton = this.createUseButton(config, this.attachModelConfig.bind(this));
        const deleteButton = this.createDeleteButton(config);

        header.appendChild(content);
        header.appendChild(useButton);
        header.appendChild(deleteButton);

        return header;
    }

    createDeleteButton(config) {
        const deleteButton = document.createElement('button');
        deleteButton.className = 'delete-button';
        deleteButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 96 960 960">
                <path fill="currentColor" d="M300 796q0 26 18 43t43 17q26 0 44-17t18-43V516q0-26-18-43t-44-17q-25 0-43 17t-18 43v280Zm240 0q0 26 18 43t43 17q26 0 43-17t17-43V516q0-26-17-43t-43-17q-26 0-43 17t-18 43v280ZM180 256v-60h600v60h136v80H44v-80h136Zm136 720q-41 0-70.5-29.5T216 876V396h528v480q0 41-29.5 70.5T644 976H316Z"/>
            </svg>
        `;
        deleteButton.onclick = async (e) => {
            e.stopPropagation();
            try {
                await this.ipcService.deleteModelConfig(config.id);
                const index = this.modelConfigs.findIndex(c => c.id === config.id);
                if (index !== -1) {
                    this.modelConfigs.splice(index, 1);
                    const configElement = document.querySelector(`.model-config-item[data-id="${config.id}"]`);
                    if (configElement) {
                        configElement.remove();
                    }
                    UIManager.showNotification('Configuration deleted', 'info');
                }
            } catch (error) {
                console.error('Error deleting model config:', error);
                UIManager.showNotification('Failed to delete configuration', 'error');
            }
        };
        return deleteButton;
    }
    
    createConfigSection(config) {
        const configSection = document.createElement('div');
        configSection.className = 'model-config-section';
        configSection.style.display = 'none';
    
        // Create options from supported models
        const modelOptions = this.supportedModels
            .map(model => `<option value="${model.name}" ${config.modelName === model.name ? 'selected' : ''}>${model.displayName}</option>`)
            .join('');
        
        configSection.innerHTML = `
            <h3>Model</h3>
            <select id="model-name-${config.id}" class="model-select">
                ${modelOptions}
            </select>
            
            ${config.modelName === 'openai-compatible-server' ? `
                <h3>Server URL:</h3>
                <input type="url" id="server-url-${config.id}" value="${config.serverUrl || ''}" placeholder="http://localhost:1234/v1">
            ` : ''}
            
            <h3>Temperature</h3>
            <input type="range" id="temperature-${config.id}" class="temperature-range" min="0" max="2" step="0.1" value="${config.temperature || 0.7}">
            <span class="temperature-value">${(config.temperature || 0.7).toFixed(1)}</span>
        `;
    
        // Add event listeners as before...
        const modelSelect = configSection.querySelector(`#model-name-${config.id}`);
        if (modelSelect) {
            modelSelect.addEventListener('change', () => {
                const serverUrlInput = configSection.querySelector(`#server-url-${config.id}`);
                if (serverUrlInput) {
                    serverUrlInput.style.display = modelSelect.value === 'openai-compatible-server' ? 'block' : 'none';
                }
            });
        }
    
        const temperatureSlider = configSection.querySelector(`#temperature-${config.id}`);
        const temperatureValue = configSection.querySelector('.temperature-value');
        if (temperatureSlider && temperatureValue) {
            temperatureSlider.addEventListener('input', (e) => {
                temperatureValue.textContent = parseFloat(e.target.value).toFixed(1);
                config.temperature = parseFloat(e.target.value);
            });
        }
    
        return configSection;
    }

    attachModelConfig(config, showSuccess = true) {
        try {
            if (this.isItemAttached(config.id)) {
                UIManager.showNotification('Configuration already attached', 'info');
                return;
            }

            const div = this.createAttachmentItem(config, 'model-config', 'M');
            const removeButton = div.querySelector('.remove-file');
            removeButton.onclick = () => this.detachModelConfig(config.id, div);

            this.elements.attachedItems.appendChild(div);
            this.emit('modelConfigAttached', config);
            
            if (showSuccess) {
                UIManager.showNotification('Configuration attached', 'success');
            }

            const searchResults = this.elements.searchResults;
            const searchItem = searchResults.querySelector(`.attachment-item[data-id="${config.id}"]`);
            if (searchItem) searchItem.remove();
        } catch (error) {
            console.error('Error attaching model config:', error);
            UIManager.showNotification('Error attaching configuration', 'error');
        }
    }

    detachModelConfig(configId, element) {
        try {
            element.remove();
            this.emit('modelConfigDetached', configId);
            this.refreshSearch();
        } catch (error) {
            console.error('Error detaching model config:', error);
            UIManager.showNotification('Error removing configuration', 'error');
        }
    }

    populateModelConfigs() {
        this.elements.configsList.innerHTML = '';
        this.modelConfigs.forEach(config => {
            const configDiv = this.createModelConfigItem(config);
            this.elements.configsList.appendChild(configDiv);
        });
    }

    getAttachedModelConfigs() {
        const attachedElements = this.elements.attachedItems.querySelectorAll('.attachment-item[data-type="model-config"]');
        return Array.from(attachedElements).map(element => {
            const config = this.modelConfigs.find(c => c.id === element.dataset.id);
            return config;
        }).filter(Boolean);
    }

    clearAttachments() {
        if (this.elements.attachedItems) {
            this.elements.attachedItems.innerHTML = '';
            this.emit('modelConfigsCleared');
        }
    }
}
