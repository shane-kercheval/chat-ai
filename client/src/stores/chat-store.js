import { EventEmitter } from '../utils/event-emitter.js';
import { AttachmentStore } from './attachment-store.js';

export class ChatStore extends EventEmitter {
    constructor(ipcService) {
        super();
        this.attachmentStore = AttachmentStore.getInstance();
        this.ipcService = ipcService;
        this.currentMessage = null;
        this.currentConversationId = null;
        this.conversations = [];
        this.contexts = [];
        this.prompts = [];
        this.instructions = [];
        this.totals = {
            messages: 0,
            inputTokens: 0,
            outputTokens: 0,
            inputCost: 0,
            outputCost: 0
        };
        this.setupIpcListeners();
        // loadInitialData is now called from app.js after server is ready
        // this.loadInitialData();
    }

    setupIpcListeners() {
        this.ipcService.onChatStreamResponse((event, data) => {
            if (data.conversation_id) {
                if (!this.currentConversationId || data.conversation_id !== this.currentConversationId) {
                    // Brand new conversation, set ID and refresh history
                    this.setCurrentConversation(data.conversation_id);
                }
            }
            // Handle response types
            if (data.type === 'chunk') {
                this.emit('chunk', {
                    content: data.content,
                    modelIndex: data.modelIndex,
                    requestEntryId: data.requestEntryId,
                    entryId: data.entryId
                });
            } else if (data.type === 'tool_event') {  // FIXED: Changed from data.tool_event to data.type === 'tool_event'
                this.emit('tool-event', {
                    type: data.tool_event.type,
                    iteration: data.tool_event.iteration,
                    thought: data.tool_event.thought,
                    toolName: data.tool_event.tool_name,
                    toolArgs: data.tool_event.tool_args,
                    result: data.tool_event.result,
                    modelIndex: data.modelIndex,
                    requestEntryId: data.requestEntryId,
                    entryId: data.entryId
                });
            } else if (data.type === 'summary') {
                this.updateTotals(data.summary);
                this.emit('summary', {
                    content: data.summary,
                    modelIndex: data.modelIndex
                });
            } else if (data.type === 'error') {
                console.error('Error from chat stream:', data.error_message);
                console.error('Error code:', data.error_code);
                this.emit('chat-error', {
                    content: data.error_message,
                    modelIndex: data.modelIndex,
                    requestEntryId: data.requestEntryId
                });
            } else {
                console.info('Unknown response type:', data.type);
            }
        });

        // Listen for resource additions to send to server
        this.attachmentStore.on('resources:added', async (resource) => {
            try {
                await this.ipcService.addResource(resource.path, resource.type);
            } catch (error) {
                console.error('Error adding resource:', error);
                this.emit('error', error);
            }
        });

        this.ipcService.onChatError((event, error) => {
            this.emit('error', error);
        });

        this.ipcService.onChatCancelled(() => {
            this.emit('cancelled');
        });
    }

    async loadInitialData() {
        try {
            // Load all data in parallel
            const [conversations, prompts, contexts, instructions, modelConfigs] = await Promise.all([
                this.ipcService.getConversations(),
                this.ipcService.getPrompts(),
                this.ipcService.getContexts(),
                this.ipcService.getInstructions(),
                this.ipcService.getModelConfigs()
            ]);
    
            this.conversations = conversations;
            this.prompts = prompts;
            this.contexts = contexts;
            this.instructions = instructions;
            this.modelConfigs = modelConfigs.configsList.map(config => ({
                id: config.configId,
                name: config.configName,
                modelType: config.config.modelType,
                modelName: config.config.modelName,
                temperature: config.config.modelParameters?.temperature || 0.7,
                serverUrl: config.config.modelParameters?.serverUrl
            }));
            this.emit('dataLoaded');
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.emit('error', error);
        }
    }

    async sendMessage(content, modelConfigs) {
        this.emit('sendStart');
        try {
            const attachedContexts = this.attachmentStore.getAttachments('contexts');

            // Get currently attached instructions
            const attachedInstructions = this.attachmentStore.getAttachments('instructions');
            const instructionTexts = attachedInstructions.map(instruction => instruction.text);
            // Get custom instructions from attached contexts
            const contextInstructions = attachedContexts
                .map(context => context.customInstructions)
                .filter(instruction => instruction?.trim()); // Filter out empty instructions
            const allInstructions = [...instructionTexts, ...contextInstructions];

            // Get resources attached in the chat window / file drop manager
            const attachedResources = this.attachmentStore.getAttachments('resources');
            // Get resources from attached contexts
            // This should get the resources dynamically attached to the current context since
            // we stored a reference to the context object, not a copy. Any modifications to the
            // context's resources array after attachment will be reflected when we access it here,
            // at send time.
            const contextResources = attachedContexts.flatMap(context => context.resources);
            // Combine all resources
            const allResources = [...attachedResources, ...contextResources];
            const contextStrategy = this.attachmentStore.getContextStrategy();
            const toolsEnabled = this.attachmentStore.getToolsEnabled(); // Add this line
            
            await this.ipcService.sendMessage(
                content, 
                modelConfigs, 
                this.currentConversationId,
                allInstructions,
                allResources,
                contextStrategy,
                toolsEnabled
            );
        } catch (error) {
            this.emit('error', error);
        }
    }

    async refreshHistory() {
        try {
            const history = await this.ipcService.getConversations();
            if (!history) {
                console.error('No history received from server');
                return;
            }
            this.conversations = history;
            this.emit('historyUpdated', this.conversations);
        } catch (error) {
            console.error('Error refreshing history:', error);
        }
    }

    async truncateConversation(entryId) {
        try {
            await this.ipcService.truncateConversation(this.currentConversationId, entryId);
            await this.refreshHistory();
            this.emit('conversationUpdated', this.currentConversationId);
        } catch (error) {
            console.error('Error truncating conversation:', error);
            this.emit('error', error);
            throw error;
        }
    }

    async branchConversation(entryId, modelIndex) {
        try {
            const response = await this.ipcService.branchConversation(
                this.currentConversationId, 
                entryId, 
                modelIndex
            );
            await this.refreshHistory();
            // Switch to new conversation
            this.setCurrentConversation(response.newConversationId);
            this.emit('conversationUpdated', response.newConversationId);
        } catch (error) {
            console.error('Error branching conversation:', error);
            this.emit('error', error);
            throw error;
        }
    }

    async selectModelResponse(entryId, modelIndex) {
        try {
            await this.ipcService.selectModelResponse(
                this.currentConversationId, 
                entryId, 
                modelIndex
            );
            await this.refreshHistory();
            this.emit('conversationUpdated', this.currentConversationId);
        } catch (error) {
            console.error('Error selecting model response:', error);
            this.emit('error', error);
            throw error;
        }
    }

    setCurrentConversation(conversationId) {
        this.currentConversationId = conversationId;
        this.refreshHistory();
    }

    getCurrentConversation() {
        if (!this.currentConversationId) return null;
        return this.conversations.find(c => c.id === this.currentConversationId);
    }

    async deleteConversation(conversationId) {
        try {
            await this.ipcService.deleteConversation(conversationId);
            // If the deleted conversation was the current one, clear it
            if (this.currentConversationId === conversationId) {
                this.currentConversationId = null;
            }
            await this.refreshHistory();
        } catch (error) {
            console.error('Error deleting conversation:', error);
            this.emit('error', error);
            throw error;
        }
    }

    updateTotals(summary) {
        this.totals.messages++;
        this.totals.inputTokens += summary.input_tokens;
        this.totals.outputTokens += summary.output_tokens;
        this.totals.inputCost += summary.input_cost;
        this.totals.outputCost += summary.output_cost;
        this.emit('totalsUpdated', this.totals);
    }
}