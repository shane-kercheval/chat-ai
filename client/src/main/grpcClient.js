const grpc = require('@grpc/grpc-js');
const {
    CompletionServiceClient,
    ContextServiceClient,
    ConfigurationServiceClient
} = require('../../proto/generated/chat_grpc_pb');
const { Empty } = require('google-protobuf/google/protobuf/empty_pb');
const { 
    ChatRequest, 
    ChatMessage,
    ModelConfig,
    ModelParameters,
    AddResourceRequest,
    Resource,
    ResourceType,
    ContextStrategy,
    EventStreamRequest,
    DeleteConversationRequest,
    SaveModelConfigRequest,
    UserModelConfig,
    DeleteModelConfigRequest,
    TruncateConversationRequest,
    BranchConversationRequest,
    MultiResponseSelectRequest,
} = require('../../proto/generated/chat_pb');

const completion_client = new CompletionServiceClient('localhost:50051', grpc.credentials.createInsecure());
const context_client = new ContextServiceClient('localhost:50051', grpc.credentials.createInsecure());
const config_client = new ConfigurationServiceClient('localhost:50051', grpc.credentials.createInsecure());

function sendChatRequest(conversationId, modelConfigs, messageText, instructions = [], resources = [], contextStrategy = null) {
    // Create the chat message
    const chatMessage = new ChatMessage();
    chatMessage.setRole(1); // USER role
    chatMessage.setContent(messageText);

    // Create the chat request
    const request = new ChatRequest();
    request.setConversationId(conversationId);
    request.setMessagesList([chatMessage]);
    request.setInstructionsList(instructions);

    // Convert each model config to protobuf ModelConfig
    const protoModelConfigs = modelConfigs.map(config => {
        const modelConfig = new ModelConfig();
        modelConfig.setModelType(config.modelType);
        modelConfig.setModelName(config.modelName);

        const modelParameters = new ModelParameters();
        modelParameters.setTemperature(config.temperature);
        if (config.serverUrl) {
            modelParameters.setServerUrl(config.serverUrl);
        }
        modelConfig.setModelParameters(modelParameters);
        
        return modelConfig;
    });
    request.setModelConfigsList(protoModelConfigs);

    const chatResources = resources.map(resource => {
        const chatResource = new Resource();
        chatResource.setPath(resource.path);
        chatResource.setType(convertResourceType(resource.type));
        return chatResource;
    });
    request.setResourcesList(chatResources);
    if (contextStrategy) {
        request.setContextStrategy(convertContextStrategy(contextStrategy));
    }
    return completion_client.chat(request);
}

function addResource(path, type) {
    return new Promise((resolve, reject) => {
        const request = new AddResourceRequest();
        request.setPath(path);
        request.setType(convertResourceType(type));
        context_client.add_resource(request, (error, response) => {
            if (error) {
                console.error('Error adding resource:', error);
                reject(error);
                return;
            }
            resolve(response);
        });
    });
}

function convertResourceType(type) {
    switch(type.toUpperCase()) {
        case 'FILE':
            return ResourceType.FILE;
        case 'DIRECTORY':
            return ResourceType.DIRECTORY;
        case 'WEBPAGE':
            return ResourceType.WEBPAGE;
        default:
            throw new Error(`Unknown resource type: ${type}`);
    }
}

function convertContextStrategy(strategy) {
    switch(strategy.toUpperCase()) {
        case 'AUTO':
            return ContextStrategy.AUTO;
        case 'RAG':
            return ContextStrategy.RAG;
        case 'FULL TEXT':
            return ContextStrategy.FULL_TEXT;
        default:
            throw new Error(`Unknown context strategy: ${strategy}`);
    }
}

function streamEvents() {
    const request = new EventStreamRequest();
    return completion_client.streamEvents(request);
}

function getHistory() {
    return new Promise((resolve, reject) => {
        const empty = new Empty();
        completion_client.get_history(empty, (error, response) => {
            if (error) {
                reject(error);
                return;
            }
            resolve(response.toObject());
        });
    });
}

function deleteConversation(conversationId) {
    return new Promise((resolve, reject) => {
        const request = new DeleteConversationRequest();
        request.setConversationId(conversationId);
        
        completion_client.delete_conversation(request, (error, response) => {
            if (error) {
                reject(error);
                return;
            }
            resolve(response);
        });
    });
}

function getSupportedModels() {
    return new Promise((resolve, reject) => {
        const empty = new Empty();
        completion_client.get_supported_models(empty, (error, response) => {
            if (error) {
                reject(error);
                return;
            }
            resolve(response.toObject());
        });
    });
}

function getModelConfigs() {
    return new Promise((resolve, reject) => {
        const empty = new Empty();
        config_client.get_model_configs(empty, (error, response) => {
            if (error) {
                reject(error);
                return;
            }
            resolve(response.toObject());
        });
    });
}

function saveModelConfig(config) {
    return new Promise((resolve, reject) => {
        const request = new SaveModelConfigRequest();
        const userModelConfig = new UserModelConfig();
        userModelConfig.setConfigName(config.name);
        
        const modelConfig = new ModelConfig();
        modelConfig.setModelType(config.modelType);
        modelConfig.setModelName(config.modelName);
        
        const modelParams = new ModelParameters();
        modelParams.setTemperature(config.temperature);
        if (config.serverUrl) {
            modelParams.setServerUrl(config.serverUrl);
        }
        modelConfig.setModelParameters(modelParams);
        
        userModelConfig.setConfig(modelConfig);
        request.setConfig(userModelConfig);
        
        config_client.save_model_config(request, (error, response) => {
            if (error) {
                reject(error);
                return;
            }
            resolve(response.toObject());
        });
    });
}

function deleteModelConfig(configId) {
    return new Promise((resolve, reject) => {
        const request = new DeleteModelConfigRequest();
        request.setConfigId(configId);
        
        config_client.delete_model_config(request, (error, response) => {
            if (error) {
                reject(error);
                return;
            }
            resolve(response);
        });
    });
}

function truncateConversation(conversationId, entryId) {
    return new Promise((resolve, reject) => {
        const request = new TruncateConversationRequest();
        request.setConversationId(conversationId);
        request.setEntryId(entryId);
        
        completion_client.truncate_conversation(request, (error, response) => {
            if (error) {
                reject(error);
                return;
            }
            resolve(response);
        });
    });
}

function branchConversation(conversationId, entryId, modelIndex) {
    return new Promise((resolve, reject) => {
        const request = new BranchConversationRequest();
        request.setConversationId(conversationId);
        request.setEntryId(entryId);
        if (modelIndex !== undefined) {
            request.setModelIndex(modelIndex);
        }
        
        completion_client.branch_conversation(request, (error, response) => {
            if (error) {
                reject(error);
                return;
            }
            resolve(response.toObject());
        });
    });
}

function multiResponseSelect(conversationId, entryId, selectedModelIndex) {
    return new Promise((resolve, reject) => {
        const request = new MultiResponseSelectRequest();
        request.setConversationId(conversationId);
        request.setEntryId(entryId);
        request.setSelectedModelIndex(selectedModelIndex);
        completion_client.multi_response_select(request, (error, response) => {
            if (error) {
                reject(error);
                return;
            }
            resolve(response);
        });
    });
}

module.exports = { 
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
    multiResponseSelect
};
