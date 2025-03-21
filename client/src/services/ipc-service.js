export const devData = {

      contexts: [
        {
            id: 'context-1',
            name: 'My Context',
            type: 'general',
            customInstructions: '',
            resources: []
        },
        {
            id: 'context-2',
            name: 'Coding Project - Server',
            type: 'coding',
            customInstructions: '',
            resources: []
        },
        {
            id: 'context-3',
            name: 'Coding Project - Client',
            type: 'coding',
            customInstructions: '',
            resources: []
        },
    ],
};

export class IpcService {
    constructor() {
        this.ipcRenderer = window.require('electron').ipcRenderer;
    }

    async getConversations() {
        try {
            const history = await this.ipcRenderer.invoke('get-history');
            return history.conversationsList.map(conv => ({
                id: conv.conversationId,
                entries: conv.entriesList,
                // messages: this.transformMessages(conv.entriesList)
            }));
        } catch (error) {
            console.error('Error fetching conversations:', error);
            return [];
        }
    }

    async deleteConversation(conversationId) {
        return this.ipcRenderer.invoke('delete-conversation', conversationId);
    }

    async getSupportedModels() {
        return this.ipcRenderer.invoke('get-supported-models');
    }

    async getPrompts() {
        try {
            const yaml = require('js-yaml');
            const fs = require('fs');
            const path = require('path');

            // Load directly from artifacts/prompts.yaml
            const promptsPath = path.join(__dirname, '../../artifacts/prompts.yaml');
            // Check if file exists
            if (!fs.existsSync(promptsPath)) {
                console.error('prompts.yaml not found, using default prompts');
                return null;
            }
            // Load and parse YAML
            const fileContents = fs.readFileSync(promptsPath, 'utf8');
            const data = yaml.load(fileContents);
            
            return data.prompts || [];
        } catch (error) {
            console.error('Error loading prompts from YAML:', error);
            return null;
        }
    }

    async getContexts() {
        try {
            // Try to get contexts from localStorage first
            const savedContexts = localStorage.getItem('contexts');
            if (savedContexts) {
                return JSON.parse(savedContexts);
            }
            // Fall back to devData if no saved contexts
            return devData.contexts;
        } catch (error) {
            console.error('Error getting contexts:', error);
        }
    }

    async getInstructions() {
        try {
            const yaml = require('js-yaml');
            const fs = require('fs');
            const path = require('path');

            // Load directly from artifacts/instructions.yaml
            const instructionsPath = path.join(__dirname, '../../artifacts/instructions.yaml');
            
            // Check if file exists
            if (!fs.existsSync(instructionsPath)) {
                console.warn('instructions.yaml not found, using default instructions');
                return devData.instructions;
            }

            // Load and parse YAML
            const fileContents = fs.readFileSync(instructionsPath, 'utf8');
            const data = yaml.load(fileContents);
            
            return data.instructions || [];
        } catch (error) {
            console.error('Error loading instructions from YAML:', error);
            // Fallback to hardcoded instructions if there's an error
            return devData.instructions;
        }
    }

    async sendMessage(
            messageText,
            modelConfigs,
            conversationId = null,
            instructions = [],
            resources = [],
            contextStrategy = null,
            enableTools = false)
    {
        return this.ipcRenderer.invoke('send-message',
            messageText,
            modelConfigs,
            conversationId,
            instructions,
            resources,
            contextStrategy,
            enableTools
        );
    }

    async addResource(path, type) {
        return this.ipcRenderer.invoke('add-resource', path, type);
    }

    cancelStream() {
        return this.ipcRenderer.invoke('cancel-stream');
    }

    async invoke(channel, data) {
        return this.ipcRenderer.invoke(channel, data);
    }

    onChatStreamResponse(callback) {
        this.ipcRenderer.on('chat-response', callback);
    }

    onChatError(callback) {
        this.ipcRenderer.on('error', callback);
    }

    onChatCancelled(callback) {
        this.ipcRenderer.on('chat-cancelled', callback);
    }

    onLoadingStart(callback) {
        this.ipcRenderer.on('loading-start', callback);
    }

    onLoadingEnd(callback) {
        this.ipcRenderer.on('loading-end', callback);
    }

    async getModelConfigs() {
        return this.ipcRenderer.invoke('get-model-configs');
    }
    
    async saveModelConfig(config) {
        return this.ipcRenderer.invoke('save-model-config', config);
    }
    
    async deleteModelConfig(configId) {
        return this.ipcRenderer.invoke('delete-model-config', configId);
    }

    async truncateConversation(conversationId, entryId) {
        return this.ipcRenderer.invoke('truncate-conversation', conversationId, entryId);
    }

    async branchConversation(conversationId, entryId, modelIndex) {
        return this.ipcRenderer.invoke('branch-conversation', conversationId, entryId, modelIndex);
    }

    async selectModelResponse(conversationId, entryId, selectedModelIndex) {
        return this.ipcRenderer.invoke('multi-response-select', conversationId, entryId, selectedModelIndex);
    }
    
}
