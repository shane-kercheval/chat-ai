import { EventEmitter } from '../utils/event-emitter.js';

export class HistoryManager extends EventEmitter {
    constructor(chatStore, chatView, eventBus) {
        super();
        this.chatStore = chatStore;
        this.chatView = chatView;
        this.eventBus = eventBus;
        this.elements = {
            historyList: document.getElementById('history-list'),
            newChatButton: document.getElementById('new-chat-button'),
            chatDiv: document.getElementById('chat')
        };
    
        // Listen for history updates
        this.chatStore.on('historyUpdated', () => {
            this.populateHistorySidebar();
        });
        
        this.initialize();
    }

    initialize() {
        this.setupNewChatHandler();
        this.populateHistorySidebar();
    }

    setupNewChatHandler() {
        if (this.elements.newChatButton) {
            this.elements.newChatButton.addEventListener('click', () => {
                this.startNewChat();
            });
        }
    }

    populateHistorySidebar() {
        this.elements.historyList.innerHTML = '';
        
        // Sort conversations by latest message timestamp
        const sortedConversations = [...this.chatStore.conversations].sort((a, b) => {
            const aLastEntry = a.entries[a.entries.length - 1];
            const bLastEntry = b.entries[b.entries.length - 1];
            
            // Convert proto timestamps to JS Date objects for comparison
            const aTime = aLastEntry ? aLastEntry.timestamp.seconds * 1000 : 0;
            const bTime = bLastEntry ? bLastEntry.timestamp.seconds * 1000 : 0;
            
            // Sort in descending order (most recent first)
            return bTime - aTime;
        });
    
        sortedConversations.forEach((conversation) => {
            const conversationDiv = this.createConversationItem(conversation);
            this.elements.historyList.appendChild(conversationDiv);
        });
    }

    createConversationItem(conversation) {
        const conversationDiv = document.createElement('div');
        conversationDiv.className = 'history-item';
        conversationDiv.dataset.id = conversation.id;

        // Add selected class if this is the current conversation
        if (conversation.id === this.chatStore.currentConversationId) {
            conversationDiv.classList.add('selected');
        }
    
        const textContainer = document.createElement('div');
        textContainer.className = 'history-item-content';
        textContainer.textContent = this.generateSummary(conversation.entries);
        
        const deleteButton = this.createDeleteButton();
        deleteButton.addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteConversation(conversation.id);
        });
    
        conversationDiv.appendChild(textContainer);
        conversationDiv.appendChild(deleteButton);
    
        conversationDiv.addEventListener('click', () => {
            this.selectConversation(conversation.id);
        });
    
        return conversationDiv;
    }

    async deleteConversation(conversationId) {
        try {
            await this.chatStore.deleteConversation(conversationId);
            // if current conversation id is null (i.e. we deleted the current conversation) then 
            // start a new chat
            if (this.chatStore.currentConversationId === null) {
                this.startNewChat();
            }
        } catch (error) {
            console.error('Error deleting conversation:', error);
        }
    }

    createDeleteButton() {
        const deleteButton = document.createElement('button');
        deleteButton.className = 'delete-button';
        deleteButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 96 960 960" style="width: 16px; height: 16px;">
                <path fill="currentColor" d="M300 796q0 26 18 43t43 17q26 0 44-17t18-43V516q0-26-18-43t-44-17q-25 0-43 17t-18 43v280Zm240 0q0 26 18 43t43 17q26 0 43-17t17-43V516q0-26-17-43t-43-17q-26 0-43 17t-18 43v280ZM180 256v-60h600v60h136v80H44v-80h136Zm136 720q-41 0-70.5-29.5T216 876V396h528v480q0 41-29.5 70.5T644 976H316Z"/>
            </svg>
        `;
        return deleteButton;
    }

    startNewChat() {
        // Clear current view
        this.chatView.clearMessages();
        // Clear current conversation ID in store
        this.chatStore.setCurrentConversation(null);
        // Clear input if any
        if (this.chatView.elements.input) {
            this.chatView.elements.input.value = '';
        }
        // Update UI selection state
        document.querySelectorAll('.history-item').forEach(item => {
            item.classList.remove('selected');
        });
    }
    
    selectConversation(conversationId) {
        const conversation = this.chatStore.conversations.find(c => c.id === conversationId);
        if (!conversation) {
            console.error('Conversation not found:', conversationId);
            return;
        }

        // Update UI selection state
        document.querySelectorAll('.history-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.id === conversationId);
        });

        this.chatView.displayConversation(conversation);
        this.chatStore.setCurrentConversation(conversationId);
    }

    generateSummary(entries) {
        // Get first user message as summary, or default text
        const firstUserMessage = entries.find(entry => 
            entry.chatMessage && entry.chatMessage.role === 1 // USER role
        );
        return firstUserMessage ? 
            firstUserMessage.chatMessage.content.slice(0, 200): 
            'New conversation';
    }
}
