const { ipcMain } = require('electron');
const {
    sendChatRequest,
    addResource,
    streamEvents,
    getHistory,
    deleteConversation,
    getSupportedModels,
    getModelConfigs,
    saveModelConfig,
    deleteModelConfig,
    truncateConversation,
    branchConversation,
    multiResponseSelect,
} = require('./grpcClient');
const grpc = require('@grpc/grpc-js');

let activeCall = null;

function setupIpcHandlers(mainWindow) {
    ipcMain.handle('send-message', async (event, messageText, modelConfigs, conversationId, instructions, resources, contextStrategy, enableTools) => {
        try {
            // Clean up any existing call
            if (activeCall) {
                activeCall.cancel();
                activeCall = null;
            }
    
            // Pass the array of model configurations to `sendChatRequest`
            const call = sendChatRequest(conversationId,
                modelConfigs,
                messageText,
                instructions,
                resources,
                contextStrategy,
                enableTools
            );
            activeCall = call;
    
            // Handle streaming data
            call.on('data', (response) => {
                if (response.hasChunk()) {
                    const chunk = response.getChunk();
                    mainWindow.webContents.send('chat-response', {
                        type: 'chunk',
                        content: chunk.getContent(),
                        logprob: chunk.getLogprob(),
                        modelIndex: response.getModelIndex(),
                        requestEntryId: response.getRequestEntryId(),
                        entryId: response.getEntryId(),
                    });
                } else if (response.hasToolEvent()) {
                    const toolEvent = response.getToolEvent();
                    mainWindow.webContents.send('chat-response', {
                        type: 'tool_event',
                        tool_event: {
                            type: toolEvent.getType(),
                            iteration: toolEvent.getIteration(),
                            thought: toolEvent.getThought(),
                            tool_name: toolEvent.getToolName(),
                            tool_args: toolEvent.getToolArgs ? 
                                Object.fromEntries(toolEvent.getToolArgs().entries()) : 
                                {},
                            result: toolEvent.getResult()
                        },
                        modelIndex: response.getModelIndex(),
                        requestEntryId: response.getRequestEntryId(),
                        entryId: response.getEntryId(),
                        conversation_id: response.getConversationId()
                    });
                } else if (response.hasSummary()) {
                    const summary = response.getSummary();
                    mainWindow.webContents.send('chat-response', {
                        type: 'summary',
                        conversation_id: response.getConversationId(),
                        summary: {
                            input_tokens: summary.getInputTokens(),
                            output_tokens: summary.getOutputTokens(),
                            input_cost: summary.getInputCost(),
                            output_cost: summary.getOutputCost(),
                            duration_seconds: summary.getDurationSeconds(),
                        },
                        modelIndex: response.getModelIndex(),
                    });
                } else if (response.hasError()) {
                    console.error('Error from server:', response.getError().getErrorMessage());
                    mainWindow.webContents.send('chat-response', {
                        type: 'error',
                        error_message: response.getError().getErrorMessage(),
                        error_code: response.getError().getErrorCode(),
                        modelIndex: response.getModelIndex(),
                        requestEntryId: response.getRequestEntryId(),
                        entryId: response.getEntryId(),
                    });
                } else {
                    console.error('Unexpected response:', response);
                    mainWindow.webContents.send('error', 'Unexpected response from server');
                }
            });
    
            // Handle errors
            call.on('error', (error) => {
                if (error.code === grpc.status.CANCELLED) {
                    // Don't propagate cancellation as an error to the UI
                    mainWindow.webContents.send('chat-cancelled');
                } else {
                    console.error('Stream error:', error);
                    mainWindow.webContents.send('error', error.message);
                }
                activeCall = null;
            });

            // Handle stream end
            call.on('end', () => {
                activeCall = null;
                mainWindow.webContents.send('chat-end');
            });

            // Handle stream close
            call.on('close', () => {
                activeCall = null;
            });

        } catch (error) {
            console.error('Error in send-message:', error);
            mainWindow.webContents.send('error', error.message);
            activeCall = null;
        }
    });

    ipcMain.handle('add-resource', async (event, path, type) => {
        try {
            return await addResource(path, type);
        } catch (error) {
            console.error('Error adding resource:', error);
            throw error;
        }
    });

    ipcMain.handle('cancel-stream', () => {
        if (activeCall) {
            try {
                activeCall.cancel();
                mainWindow.webContents.send('chat-cancelled');
            } catch (error) {
                console.error('Error during cancellation:', error);
                mainWindow.webContents.send('error', 'Failed to cancel stream');
            } finally {
                activeCall = null;
            }
        } else {
            console.log('No active stream to cancel');
        }
    });

    ipcMain.on('start-event-stream', async () => {
        try {
            const call = streamEvents();
            
            call.on('data', (event) => {
                mainWindow.webContents.send('server-event', {
                    type: event.getType(),
                    level: event.getLevel(),
                    timestamp: event.getTimestamp()?.toDate(),
                    message: event.getMessage(),
                    metadata: Object.fromEntries(event.getMetadataMap().entries())
                });
            });

            call.on('error', (error) => {
                console.error('Event stream error:', error);
                mainWindow.webContents.send('server-event', {
                    type: 'SERVICE',
                    level: 'ERROR',
                    timestamp: new Date(),
                    message: 'Event stream error: ' + error.message,
                    metadata: {}
                });
            });

        } catch (error) {
            console.error('Failed to start event stream:', error);
            mainWindow.webContents.send('server-event', {
                type: 'SERVICE',
                level: 'ERROR',
                timestamp: new Date(),
                message: 'Failed to start event stream: ' + error.message,
                metadata: {}
            });
        }
    });

    ipcMain.handle('get-history', async () => {
        try {
            return await getHistory();
        } catch (error) {
            console.error('Error getting history:', error);
            throw error;
        }
    });

    ipcMain.handle('delete-conversation', async (event, conversationId) => {
        try {
            await deleteConversation(conversationId);
        } catch (error) {
            console.error('Error deleting conversation:', error);
            throw error;
        }
    });

    ipcMain.handle('get-supported-models', async () => {
        try {
            return await getSupportedModels();
        } catch (error) {
            console.error('Error getting supported models:', error);
            throw error;
        }
    });

    ipcMain.handle('get-model-configs', async () => {
        try {
            return await getModelConfigs();
        } catch (error) {
            console.error('Error getting model configs:', error);
            throw error;
        }
    });
    
    ipcMain.handle('save-model-config', async (event, config) => {
        try {
            return await saveModelConfig(config);
        } catch (error) {
            console.error('Error saving model config:', error);
            throw error;
        }
    });
    
    ipcMain.handle('delete-model-config', async (event, configId) => {
        try {
            return await deleteModelConfig(configId);
        } catch (error) {
            console.error('Error deleting model config:', error);
            throw error;
        }
    });

    ipcMain.handle('truncate-conversation', async (event, conversationId, entryId) => {
        try {
            await truncateConversation(conversationId, entryId);
        } catch (error) {
            console.error('Error truncating conversation:', error);
            throw error;
        }
    });

    ipcMain.handle('branch-conversation', async (event, conversationId, entryId, modelIndex) => {
        try {
            return await branchConversation(conversationId, entryId, modelIndex);
        } catch (error) {
            console.error('Error branching conversation:', error);
            throw error;
        }
    });

    ipcMain.handle('multi-response-select', async (event, conversation_id, entry_id, selected_model_index) => {
        try {
            return await multiResponseSelect(conversation_id, entry_id, selected_model_index);
        } catch (error) {
            console.error('Error selecting model response:', error);
            throw error;
        }
    });
}

module.exports = { setupIpcHandlers };