import { IpcService } from './services/ipc-service.js';
import { AttachmentStore } from './stores/attachment-store.js';
import { ChatStore } from './stores/chat-store.js';
import { ChatView } from './components/chat-view.js';
import { ContextManager } from './components/context-manager.js';
import { PromptManager } from './components/prompt-manager.js';
import { InstructionManager } from './components/instruction-manager.js';
import { FileDropManager } from './components/file-drop-manager.js';
import { SidebarManager } from './components/sidebar-manager.js';
import { UIManager } from './utils/ui-utils.js';
import { AppEventBus } from './utils/app-event-bus.js';
import { HistoryManager } from './components/history-manager.js';
import { EventManager } from './components/event-manager.js';
import { ModelConfigManager } from './components/model-config-manager.js';

class ChatApp {
    constructor() {
        this.eventBus = AppEventBus.getInstance();
        this.initializeServices();
        this.initializeStores();
        // Listen for data loaded event before initializing components
        this.chatStore.on('dataLoaded', () => {
            this.initializeComponents();
            this.setupKeyboardShortcuts();
            this.setupEventHandlers();
        });
    }

    initializeServices() {
        this.ipcService = new IpcService();
    }

    initializeStores() {
        this.attachmentStore = new AttachmentStore();
        this.chatStore = new ChatStore(this.ipcService);
    }

    initializeComponents() {
        this.sidebarManager = new SidebarManager();

        this.modelConfigManager = new ModelConfigManager(
            this.chatStore.modelConfigs,
            document.getElementById('models-list'),
            this.ipcService
        );

        this.contextManager = new ContextManager(
            this.chatStore.contexts,
            document.getElementById('contexts-list'),
        );

        this.instructionManager = new InstructionManager(
            this.chatStore.instructions,
            document.getElementById('instructions-list'),
        );

        this.chatView = new ChatView(
            this.chatStore,
            this.modelConfigManager,
            {
                chatDiv: 'chat',
                input: 'message-input',
                sendButton: 'send-button',
                cancelButton: 'cancel-button',
            },
        );

        this.promptManager = new PromptManager(
            this.chatStore.prompts
        );

        this.historyManager = new HistoryManager(
            this.chatStore,
            this.chatView,
            this.eventBus
        );

        this.fileDropManager = new FileDropManager();

        new EventManager();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Check for Cmd+N (Mac) or Ctrl+N (Windows/Linux)
            if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
                e.preventDefault();  // Prevent default browser behavior
                this.historyManager.startNewChat();
            }
            // Check for Cmd+B (Mac) or Ctrl+B (Windows/Linux)
            if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
                e.preventDefault();  // Prevent default browser behavior
                this.sidebarManager.toggleSidebar();
            }
        });
    }

    setupEventHandlers() {

        // Chat-related events
        this.eventBus.on('message:send', async (data) => {
            try {
                UIManager.showLoading(document.getElementById('chat'));
                await this.chatStore.sendMessage(data.content, data.modelConfigs);
            } catch (error) {
                UIManager.showNotification('Failed to send message', 'error');
            }
        });

        this.contextManager.on('resourceAdded', (resource) => {
            this.ipcService.addResource(resource.path, resource.type);
        });

        // Prompt-related events
        this.promptManager.on('promptUsed', (prompt) => {
            this.eventBus.emit('prompt:used', prompt);
        });

        // Handle global errors
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            UIManager.showNotification('An unexpected error occurred', 'error');
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.app = new ChatApp();
    } catch (error) {
        console.error('Error initializing app:', error);
        UIManager.showNotification('Failed to initialize application', 'error');
    }
});
