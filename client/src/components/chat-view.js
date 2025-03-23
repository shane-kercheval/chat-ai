const marked = require('marked');
const hljs = require('highlight.js');
const renderMathInElement = require('katex/dist/contrib/auto-render.js');
import { UIManager } from '../utils/ui-utils.js';

const copy_button_svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
        <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
        <line x1="8" y1="10" x2="16" y2="10"></line>
        <line x1="8" y1="14" x2="16" y2="14"></line>
        <line x1="8" y1="18" x2="12" y2="18"></line>
    </svg>
`;

const delete_button_svg = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
    </svg>
`;

const branch_button_svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="6" y1="3" x2="6" y2="15"></line>
        <circle cx="18" cy="6" r="3"></circle>
        <circle cx="6" cy="18" r="3"></circle>
        <path d="M18 9a9 9 0 0 1-9 9"></path>
    </svg>
`;

const select_button_svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather">
        <path d="M20 6L9 17l-5-5"></path>
    </svg>
`;

export class ChatView {
    constructor(chatStore, modelConfigManager, elementSelectors) {
        this.store = chatStore;
        this.elements = {};
        this.modelConfigManager = modelConfigManager;
        this.currentMessages = new Map();
        this.receivedSummaries = new Set();
        this.isManualScrolling = false;
        this.scrollThreshold = 5; // pixels from bottom to consider "at bottom"
        
        this.loadEmptyStateInstructions().then(() => {
            this.initializeElements(elementSelectors);
            this.initializeEventListeners();
            this.setupStoreCallbacks();
            this.checkEmptyState();
        });
    }

    async loadEmptyStateInstructions() {
        try {
            const response = await fetch('./assets/empty-chat-instructions.md');
            if (!response.ok) throw new Error('Failed to load instructions');
            this.emptyStateInstructions = await response.text();
        } catch (error) {
            console.error('Error loading empty state instructions:', error);
            this.emptyStateInstructions = '# Welcome to Chat\n\nStart a new conversation...';
        }
    }

    checkEmptyState() {
        if (!this.elements.chatDiv) {
            console.error('Chat div not initialized');
            return;
        }
    
        // Remove any existing empty state instructions first
        const existingInstructions = this.elements.chatDiv.querySelector('.empty-state-instructions');
        if (existingInstructions) {
            existingInstructions.remove();
        }
    
        // Check if chat is truly empty (no messages)
        const hasMessages = this.elements.chatDiv.querySelector('.message');
        if (!hasMessages) {
            const instructionsDiv = document.createElement('div');
            instructionsDiv.className = 'empty-state-instructions';
            instructionsDiv.innerHTML = marked.parse(this.emptyStateInstructions);
            this.elements.chatDiv.appendChild(instructionsDiv);
        }
    }

    displayConversation(conversation) {
        this.clearMessages();
        
        conversation.entries.forEach(entry => {
            if (entry.chatMessage) {
                if (entry.chatMessage.role === 1) { // USER role
                    this.addUserMessage(entry.chatMessage.content, entry.entryId);
                } else if (entry.chatMessage.role === 2) { // ASSISTANT role
                    this.addAssistantMessage([{}]);
                    this.appendToCurrentMessage(entry.chatMessage.content, 0);
                }
            } else if (entry.singleModelResponse) {
                this.addAssistantMessage([entry.singleModelResponse.configSnapshot]);
                const messageDiv = this.appendToCurrentMessage(
                    entry.singleModelResponse.message.content,
                    0,
                    entry.entryId
                );
                messageDiv.classList.add('response-complete');
            } else if (entry.multiModelResponse) {
                const sortedResponses = entry.multiModelResponse.responsesList.slice().sort(
                    (a, b) => a.modelIndex - b.modelIndex
                );
                const configs = sortedResponses.map(response => response.configSnapshot);
                this.addAssistantMessage(configs);
                sortedResponses.forEach((response, index) => {
                    const messageDiv = this.appendToCurrentMessage(
                        response.message.content,
                        index,
                        entry.entryId
                    );
                    messageDiv.classList.add('response-complete');
                    if (entry.multiModelResponse.selectedModelIndex !== undefined && 
                            entry.multiModelResponse.selectedModelIndex.value === response.modelIndex) {
                        messageDiv.classList.add('selected');
                        const selectButton = messageDiv.querySelector('.select-button');
                        if (selectButton) {
                            selectButton.classList.add('selected');
                        }
                    }
                });
            }
        });
    }

    clearMessages() {
        this.elements.chatDiv.innerHTML = '';
        this.checkEmptyState();
    }

    initializeElements(selectors) {
        this.elements = {
            chatDiv: document.getElementById('chat'),
            input: document.getElementById('message-input'),
            sendButton: document.getElementById('send-button'),
            cancelButton: document.getElementById('cancel-button'),
            modelSelect: document.getElementById('model-select'),
            serverUrlInput: document.getElementById('server-url'),
            serverUrlContainer: document.getElementById('server-url-container'),
            temperatureSlider: document.getElementById('temperature-slider'),
            totalMessages: document.getElementById('total-messages'),
            totalInputTokens: document.getElementById('total-input-tokens'),
            totalOutputTokens: document.getElementById('total-output-tokens'),
            totalCacheWriteTokens: document.getElementById('total-cache-write-tokens'),
            totalCacheReadTokens: document.getElementById('total-cache-read-tokens'),
            totalInputCost: document.getElementById('total-input-cost'),
            totalOutputCost: document.getElementById('total-output-cost'),
            totalCacheWriteCost: document.getElementById('total-cache-write-cost'),
            totalCacheReadCost: document.getElementById('total-cache-read-cost'),
            totalCost: document.getElementById('total-cost')
        };
    }

    initializeEventListeners() {
        this.elements.sendButton.addEventListener('click', () => this.sendMessage());
        this.elements.cancelButton.addEventListener('click', () => this.store.ipcService.cancelStream());
        
        this.elements.temperatureSlider.addEventListener('input', (e) => {
            // Update the displayed value
            const temperatureValue = document.getElementById('temperature-value');
            if (temperatureValue) {
                temperatureValue.textContent = e.target.value;
            }
        });

        this.elements.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.elements.modelSelect.addEventListener('change', () => {
            if (this.elements.modelSelect.value === 'openai-compatible-server') {
                this.elements.serverUrlContainer.style.display = 'block';
            } else {
                this.elements.serverUrlContainer.style.display = 'none';
            }
        });

        // handle manual scrolling (i.e. disable auto-scroll) while response is streaming
        this.elements.chatDiv.addEventListener('scroll', () => {
            const scrollPosition = this.elements.chatDiv.scrollTop + this.elements.chatDiv.clientHeight;
            const isAtBottom = (this.elements.chatDiv.scrollHeight - scrollPosition) <= this.scrollThreshold;
            // If user scrolls to bottom, re-enable auto-scroll
            if (isAtBottom) {
                this.isManualScrolling = false;
            } else {
                this.isManualScrolling = true;
            }
        });
    }

    setupStoreCallbacks() {
        this.store.on('chunk', (data) => {
            if (data.requestEntryId) {
                this.addDeleteButtonToLastUserMessage(data.requestEntryId);
            } else {
                console.error('No request entry ID found in chunk data');
            }
            this.appendToCurrentMessage(
                data.content,
                data.modelIndex,
                data.entryId
            );
        });

        this.store.on('tool-event', (data) => {
            // Stop loading spinner on first tool event
            this.setLoading(false);
            
            if (data.requestEntryId) {
                this.addDeleteButtonToLastUserMessage(data.requestEntryId);
            }
            
            let content = '';
            switch(data.type) {
                case 0: // THINK_START
                    content = `<div><em>Thinking iteration ${data.iteration + 1}...</em></div>`;
                    break;
                case 1: // THOUGHT
                    if (!data.thought) {
                        content = `<div><em>Thinking...</em></div>`;
                        break;
                    }
                    content = [
                        '<div>',
                        `<strong>Thought:</strong>`,
                        data.thought,
                        '</div>'
                    ].join('');
            
                    if (data.toolName) {
                        content += [
                            '<div>',
                            '<strong>Tool Call:</strong>',
                            `<span class="tool-name">tool: ${data.toolName}</span><br>`,
                            '<span class="tool-label">arguments:</span>',
                            '<pre>',
                            JSON.stringify(data.toolArgs, null, 2),
                            '</pre>',
                            '</div>'
                        ].join('');
                    }
                    break;
                case 2: // TOOL_EXECUTION_START
                    content = [
                        '<div>',
                        '<strong>Tool Execution:</strong>',
                        `<span class="tool-name">tool: ${data.toolName}</span><br>`,
                        '<span class="tool-label">argument:</span>',
                        '<pre>',
                        JSON.stringify(data.toolArgs, null, 2),
                        '</pre>',
                        '</div>'
                    ].join('');
                    break;
                case 3: // TOOL_EXECUTION_RESULT
                    let resultValue = data.result;
                    try {
                        const match = data.result.match(/text='(\d+)'/);
                        if (match) {
                            resultValue = match[1];
                        }
                    } catch (e) {
                        console.error('Error parsing result:', e);
                    }
                    
                    content = [
                        '<div>',
                        '<strong>Tool Result:</strong>',
                        `<span class="tool-name">tool: ${data.toolName}</span><br>`,
                        '<span class="tool-label">result:</span>',
                        '<pre>',
                        `${resultValue}`,
                        '</pre>',
                        '</div>'
                    ].join('');
                    break;
            }

            const messageDiv = this.currentMessages.get(data.modelIndex);
            if (!messageDiv) {
                console.error('No message div found for model index:', data.modelIndex);
                return;
            }

            const toolEventsContent = messageDiv.querySelector('.tool-events-content');
            if (!toolEventsContent) {
                console.error('No tool events content div found');
                return;
            }

            // Append the tool event content
            const previousContent = toolEventsContent.innerHTML || '';
            const updatedContent = previousContent + marked.parse(content);
            toolEventsContent.innerHTML = updatedContent;

            // Show the tool events container if it was hidden
            const toolEventsContainer = messageDiv.querySelector('.tool-events-container');
            if (toolEventsContainer) {
                toolEventsContainer.style.display = 'block';
            }

            // Update syntax highlighting for code blocks
            toolEventsContent.querySelectorAll('pre code').forEach((block) => {
                // Remove the data-highlighted attribute before highlighting again
                block.removeAttribute('data-highlighted');
                hljs.highlightElement(block);
            });

            if (!this.isManualScrolling) {
                requestAnimationFrame(() => {
                    this.elements.chatDiv.scrollTop = this.elements.chatDiv.scrollHeight;
                });
            }
        });

        this.store.on('summary', (data) => {
            const messageDiv = this.currentMessages.get(data.modelIndex);
            if (messageDiv) {
                messageDiv.classList.add('response-complete');
            }
            this.receivedResponses++;
            if (this.receivedResponses === this.expectedResponses) {
                this.finalizeMessages();
                this.receivedResponses = 0;
                this.expectedResponses = null;
            }
        });

        this.store.on('chat-error', (data) => {
            // in the case of an error we still want to add the 'delete' icon
            // to the last user message
            if (data.requestEntryId) {
                this.addDeleteButtonToLastUserMessage(data.requestEntryId);
            } else {
                console.error('No request entry ID found in chunk data');
            }
            this.appendToCurrentMessage(data.content, data.modelIndex);
            UIManager.showNotification("Request Error", 'error');
        });

        this.store.on('error', (error) => {
            this.handleError(error);
            UIManager.showNotification(error, 'error');
        });

        this.store.on('cancelled', () => {
            this.handleCancelled();
            UIManager.showNotification('Request cancelled', 'warning');
        });

        this.store.on('sendStart', () => {
            this.setLoading(true);
        });

        this.store.on('totalsUpdated', (totals) => {
            this.updateSummary(totals);
        });

        this.store.on('conversationUpdated', (conversationId) => {
            const conversation = this.store.conversations.find(c => c.id === conversationId);
            if (conversation) {
                this.displayConversation(conversation);
            } else {
                console.error('Conversation not found:', conversationId);
            }
        });
    }

    setLoading(loading) {
        this.isLoading = loading;
        if (loading) {
            this.elements.sendButton.disabled = true;
            UIManager.showLoading(this.elements.chatDiv);
        } else {
            this.elements.sendButton.disabled = false;
            UIManager.hideLoading(this.elements.chatDiv);
        }
    }

    async sendMessage() {
        if (this.isLoading) return;

        // Reset manual scrolling when starting a new message
        this.isManualScrolling = false;

        const content = this.elements.input.value.trim();
        if (!content) return;

        const modelConfigs = this.getModelConfigs();
        if (!modelConfigs) return;

        try {
            this.expectedResponses = modelConfigs.length;
            this.receivedResponses = 0;
            this.addUserMessage(content);
            this.addAssistantMessage();
            // If this is a new conversation, the server will send back a conversation_id
            // in the stream response, which will be handled by the store
            await this.store.sendMessage(content, modelConfigs);
            this.elements.input.value = '';
            this.elements.input.style.height = 'auto';
        } catch (error) {
            console.error('Error sending message:', error);
            UIManager.showNotification('Failed to send message', 'error');
        }
    }

    addDeleteButtonToLastUserMessage(requestEntryId) {
        const lastUserMessage = Array.from(this.elements.chatDiv.querySelectorAll('.message.user'))
            .pop();
        if (lastUserMessage && !lastUserMessage.dataset.entryId) {
            lastUserMessage.dataset.entryId = requestEntryId;
            const deleteButton = document.createElement('button');
            deleteButton.className = 'truncate-button';
            deleteButton.innerHTML = delete_button_svg;
            deleteButton.addEventListener('click', () => this.handleTruncateMessage(requestEntryId));
            lastUserMessage.appendChild(deleteButton);
        }
    }

    getModelConfigs() {
        const attachedConfigs = this.modelConfigManager.getAttachedModelConfigs();
        if (!attachedConfigs || attachedConfigs.length === 0) {
            UIManager.showNotification('Please attach at least one model configuration', 'warning');
            return null;
        }
        return attachedConfigs;
    }

    // Update existing methods to use UIManager for user feedback
    handleError(error) {
        this.appendToCurrentMessage(`\nError: ${error}`);
        this.currentMessage = null;
        this.setLoading(false);
    }

    handleCancelled() {
        this.appendToCurrentMessage('\n\nRequest Cancelled');
        this.currentMessage = null;
        this.setLoading(false);
    }

    finalizeMessages() {
        this.currentMessages.forEach(messageDiv => {
            this.addCopyButtons(messageDiv);
        });
        this.currentMessages = new Map();
    }

    addUserMessage(content, entryId = null) {
        const emptyStateDiv = this.elements.chatDiv.querySelector('.empty-state-instructions');
        if (emptyStateDiv) {
            emptyStateDiv.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.dataset.originalContent = content;
        if (entryId) {
            messageDiv.dataset.entryId = entryId;
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = marked.parse(content);
        messageDiv.appendChild(contentDiv);

        if (entryId) {
            const deleteButton = document.createElement('button');
            deleteButton.className = 'truncate-button';
            deleteButton.innerHTML = delete_button_svg;
            deleteButton.addEventListener('click', () => this.handleTruncateMessage(entryId));
            messageDiv.appendChild(deleteButton);
        }

        this.elements.chatDiv.appendChild(messageDiv);
        messageDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        if (!this.isManualScrolling) {
            requestAnimationFrame(() => {
                this.elements.chatDiv.scrollTop = this.elements.chatDiv.scrollHeight;
            });
        }
    }

    async handleTruncateMessage(entryId) {
        try {
            await this.store.truncateConversation(entryId);
        } catch (error) {
            console.error('Error truncating conversation:', error);
            UIManager.showNotification('Failed to truncate conversation', 'error');
        }
    }

    async handleBranchMessage(messageDiv) {
        try {
            const modelIndex = parseInt(messageDiv.dataset.modelIndex);
            await this.store.branchConversation(messageDiv.dataset.entryId, modelIndex);
        } catch (error) {
            console.error('Error branching conversation:', error);
        }
    }

    async handleMultiResponseSelect(messageDiv) {
        const modelIndex = parseInt(messageDiv.dataset.modelIndex);
        const entryId = messageDiv.dataset.entryId;
        
        try {
            await this.store.selectModelResponse(entryId, modelIndex);
            
            // Update UI - change borders/svg
            const container = messageDiv.parentElement;
            container.querySelectorAll('.message.assistant').forEach(msg => {
                msg.classList.remove('selected');
                msg.querySelector('.select-button')?.classList.remove('selected');
            });
            messageDiv.classList.add('selected');
            messageDiv.querySelector('.select-button')?.classList.add('selected');
        } catch (error) {
            console.error('Error selecting response:', error);
        }
    }

    addAssistantMessage(modelConfigs = null) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';
        // Use provided configs or get from model manager
        // In the case of streaming, we can get from attached configs
        // In the case of loading from history, we need to use the provided configs
        const configs = modelConfigs || this.modelConfigManager.getAttachedModelConfigs().map(config => ({
            modelName: config.modelName,
            modelParameters: {
                temperature: config.temperature,
                serverUrl: config.serverUrl
            }
        }));
        messageContainer.style.gridTemplateColumns = `repeat(${configs.length}, 1fr)`;
        const isMultiModel = configs.length > 1;
        this.currentMessages = new Map();
    
        // Create array of message divs
        const messageDivs = configs.map((config, index) => {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            messageDiv.dataset.modelIndex = index;
            messageDiv.dataset.buffer = '';
            
            const modelLabel = document.createElement('div');
            modelLabel.className = 'model-label';
            // During streaming, ensure we have the same format as history
            const labelConfig = modelConfigs ? config : {
                modelName: config.modelName,
                modelParameters: config.modelParameters
            };
            modelLabel.textContent = generateModelLabel(labelConfig);
            messageDiv.appendChild(modelLabel);
    
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            messageDiv.appendChild(contentDiv);

            // Add tool events section with new design
            const toolEventsContainer = document.createElement('div');
            toolEventsContainer.className = 'tool-events-container';
            toolEventsContainer.style.display = 'none'; // Hide by default
            
            const toolEventsHeader = document.createElement('div');
            toolEventsHeader.className = 'tool-events-header';
            toolEventsHeader.innerHTML = `
                <div class="tool-events-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
                    </svg>
                    <span>MCP Agent Events</span>
                </div>
                <button class="tool-events-toggle">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="18 15 12 9 6 15"></polyline>
                    </svg>
                </button>
            `;
            
            const toolEventsContent = document.createElement('div');
            toolEventsContent.className = 'tool-events-content';
            
            toolEventsHeader.querySelector('.tool-events-toggle').addEventListener('click', (e) => {
                const container = toolEventsContainer;
                const button = e.currentTarget;
                const isCollapsed = container.classList.toggle('collapsed');
                button.classList.toggle('collapsed', isCollapsed);
            });
            
            toolEventsContainer.appendChild(toolEventsHeader);
            toolEventsContainer.appendChild(toolEventsContent);
            messageDiv.insertBefore(toolEventsContainer, contentDiv);
            
            // Store reference to tool events content
            messageDiv.dataset.toolEventsContent = true;

            const branchButton = document.createElement('button');
            branchButton.className = 'branch-button';
            branchButton.innerHTML = branch_button_svg;
            branchButton.addEventListener('click', () => this.handleBranchMessage(messageDiv));
            messageDiv.appendChild(branchButton);

            if (isMultiModel) {
                const selectButton = document.createElement('button');
                selectButton.className = 'select-button';
                selectButton.innerHTML = select_button_svg;
                selectButton.addEventListener('click', () => this.handleMultiResponseSelect(messageDiv));
                messageDiv.appendChild(selectButton);
            }
            
            this.currentMessages.set(index, messageDiv);
            return messageDiv;
        });
    
        // Sort and append message divs by model index
        messageDivs.sort((a, b) => Number(a.dataset.modelIndex) - Number(b.dataset.modelIndex));
        messageDivs.forEach(div => messageContainer.appendChild(div));
    
        this.elements.chatDiv.appendChild(messageContainer);
        if (!this.isManualScrolling) {
            requestAnimationFrame(() => {
                this.elements.chatDiv.scrollTop = this.elements.chatDiv.scrollHeight;
            });
        }
    }

    appendToCurrentMessage(content, modelIndex, entryId = null) {
        this.setLoading(false);
        const messageDiv = this.currentMessages.get(modelIndex);
        if (!messageDiv) {
            // request might be cancelled, so ignore
            console.log('No message div found for model index:', modelIndex);
            return;
        }

        if (entryId) {
            messageDiv.dataset.entryId = entryId;
        }
    
        const contentDiv = messageDiv.querySelector('.message-content');
        if (!contentDiv) {
            console.error('No content div found in message div');
            return;
        }
    
        const previousContent = messageDiv.dataset.buffer || '';
        const updatedContent = previousContent + content;
        
        messageDiv.dataset.buffer = updatedContent;
        messageDiv.dataset.originalContent = updatedContent;
        contentDiv.innerHTML = marked.parse(updatedContent);
    
        contentDiv.querySelectorAll('pre code').forEach((block) => {
            // Remove the data-highlighted attribute before highlighting again
            block.removeAttribute('data-highlighted');
            hljs.highlightElement(block);
        });

        try {
            renderMathInElement(contentDiv, {
                delimiters: [
                    {left: '[', right: ']', display: true},
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false}
                ],
                throwOnError: false,
                strict: false,
                trust: true,
                macros: {},
                output: 'html'
            });
        } catch (error) {
            console.error('KaTeX rendering error:', error);
        }
    
        setTimeout(() => {
            this.addCopyButtons(messageDiv);
        }, 0);
    
        // Update scroll handling to be more immediate
        if (!this.isManualScrolling) {
            requestAnimationFrame(() => {
                this.elements.chatDiv.scrollTop = this.elements.chatDiv.scrollHeight;
            });
        }
        return messageDiv;
    }

    addCopyButtons(messageDiv) {
        if (!messageDiv) return;
        this.addCodeCopyButtons(messageDiv);
        this.addMessageCopyButtons(messageDiv);
    }
    
    addCodeCopyButtons(messageDiv) {
        messageDiv.querySelectorAll('pre code').forEach((block) => {
            const pre = block.parentNode;
            
            if (!pre.querySelector('.copy-button--code')) {
                const copyButton = document.createElement('button');
                copyButton.classList.add('copy-button--code', 'copy-button');
                copyButton.innerHTML = copy_button_svg;

                copyButton.addEventListener('click', () => {
                    const code = block.innerText;
                    navigator.clipboard.writeText(code);
                    copyButton.classList.add('copied');
                    setTimeout(() => {
                        copyButton.classList.remove('copied');
                    }, 150);
                });
                
                pre.appendChild(copyButton);
            }
        });
    }
    
    addMessageCopyButtons(messageDiv) {
        if (!messageDiv.querySelector('.copy-button--message')) {
            const copyButton = document.createElement('button');
            copyButton.classList.add('copy-button--message', 'copy-button');
            copyButton.innerHTML = copy_button_svg;

            copyButton.addEventListener('click', () => {
                const currentContent = messageDiv.dataset.buffer || messageDiv.dataset.originalContent;
                navigator.clipboard.writeText(currentContent)
                    .then(() => {
                        copyButton.classList.add('copied');
                        setTimeout(() => {
                            copyButton.classList.remove('copied');
                        }, 150);
                    })
                    .catch((err) => console.error('Copy failed:', err));
            });
            messageDiv.appendChild(copyButton);
        }
    }

    updateSummary(totals) {
        this.elements.totalMessages.textContent = totals.messages;
        this.elements.totalInputTokens.textContent = totals.inputTokens;
        this.elements.totalOutputTokens.textContent = totals.outputTokens;
        this.elements.totalCacheWriteTokens.textContent = totals.cacheWriteTokens;
        this.elements.totalCacheReadTokens.textContent = totals.cacheReadTokens;
        this.elements.totalInputCost.textContent = totals.inputCost.toFixed(5);
        this.elements.totalOutputCost.textContent = totals.outputCost.toFixed(5);
        this.elements.totalCacheWriteCost.textContent = totals.cacheWriteCost.toFixed(5);
        this.elements.totalCacheReadCost.textContent = totals.cacheReadCost.toFixed(5);
        
        // Calculate total cost including cache costs
        const totalCost = totals.inputCost + totals.outputCost + 
            totals.cacheWriteCost + totals.cacheReadCost;
        this.elements.totalCost.textContent = totalCost.toFixed(5);
    }
}

function generateModelLabel(config) {
    const params = [];
    if (config.modelParameters) {
        if (config.modelParameters.temperature !== null) {
            params.push(`temp: ${config.modelParameters.temperature.toFixed(1)}`);
        }
        if (config.modelParameters.maxTokens) {
            params.push(`max tokens: ${config.modelParameters.maxTokens}`);
        }
        if (config.modelParameters.topP) {
            params.push(`top p: ${config.modelParameters.topP.toFixed(1)}`);
        }
        if (config.modelParameters.serverUrl) {
            params.push(`url: ${config.modelParameters.serverUrl}`);
        }
    }

    return `${config.modelName}${params.length ? ` (${params.join(', ')})` : ''}`;
}
