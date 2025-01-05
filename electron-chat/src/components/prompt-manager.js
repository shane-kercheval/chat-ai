import { UIManager } from '../utils/ui-utils.js';
import { EventEmitter } from '../utils/event-emitter.js';

export class PromptManager extends EventEmitter {
    constructor(prompts = []) {
        super();
        this.prompts = prompts;
        this.elements = {
            promptsList: document.getElementById('prompts-list'),
            messageInput: document.getElementById('message-input'),
            addButton: document.getElementById('add-prompt-button')
        };
        this.initialize();
    }

    initialize() {
        try {
            this.setupAddButton();
            this.populatePrompts();
        } catch (error) {
            console.error('Error initializing prompts:', error);
            UIManager.showNotification('Error loading prompts', 'error');
        }
    }

    setupAddButton() {
        this.elements.addButton.addEventListener('click', () => {
            UIManager.showNotification(
                'Not yet supported. Edit ./artifacts/prompts.yaml to add or modify prompts.',
                'warning'
            );
        });
    }

    createPromptItem(prompt) {
        try {
            const promptDiv = document.createElement('div');
            promptDiv.className = 'history-item';
            promptDiv.dataset.id = prompt.id;

            const textContainer = document.createElement('div');
            textContainer.className = 'history-item-content prompt-content';

            const title = document.createElement('div');
            title.className = 'prompt-title';
            title.textContent = prompt.description;
            title.title = prompt.description;

            const content = document.createElement('div');
            content.className = 'prompt-content-preview';
            content.textContent = prompt.template;
            content.title = prompt.template;

            textContainer.appendChild(title);
            textContainer.appendChild(content);

            const useButton = document.createElement('button');
            useButton.className = 'use-button';
            useButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
            `;
            useButton.onclick = () => this.usePrompt(prompt);

            promptDiv.appendChild(textContainer);
            promptDiv.appendChild(useButton);

            return promptDiv;
        } catch (error) {
            console.error('Error creating prompt item:', error);
            throw error;
        }
    }

    populatePrompts() {
        try {
            this.elements.promptsList.innerHTML = '';
            this.prompts.forEach(prompt => {
                const promptDiv = this.createPromptItem(prompt);
                this.elements.promptsList.appendChild(promptDiv);
            });
        } catch (error) {
            console.error('Error populating prompts:', error);
            UIManager.showNotification('Error displaying prompts', 'error');
        }
    }

    usePrompt(prompt) {
        try {
            if (this.elements.messageInput.value) {
                this.elements.messageInput.value += '\n\n';
            }
            this.elements.messageInput.value += prompt.template;
            this.elements.messageInput.focus();
            
            this.emit('promptUsed', prompt);
            UIManager.showNotification('Prompt appended to chat', 'success');
        } catch (error) {
            console.error('Error using prompt:', error);
            UIManager.showNotification('Error applying prompt', 'error');
        }
    }

    async addPrompt(promptData) {
        try {
            UIManager.showLoading(this.elements.promptsList);
            
            const prompt = {
                id: String(Date.now()),
                ...promptData
            };
            
            this.prompts.push(prompt);
            const promptDiv = this.createPromptItem(prompt);
            this.elements.promptsList.appendChild(promptDiv);
            
            this.emit('promptAdded', prompt);
            UIManager.showNotification('Prompt added successfully', 'success');
            
        } catch (error) {
            console.error('Error adding prompt:', error);
            UIManager.showNotification('Error adding prompt', 'error');
        } finally {
            UIManager.hideLoading(this.elements.promptsList);
        }
    }

    async deletePrompt(promptId) {
        try {
            const index = this.prompts.findIndex(p => p.id === promptId);
            if (index !== -1) {
                this.prompts.splice(index, 1);
                const promptDiv = this.elements.promptsList.querySelector(`[data-id="${promptId}"]`);
                if (promptDiv) {
                    promptDiv.remove();
                }
                
                this.emit('promptDeleted', promptId);
                UIManager.showNotification('Prompt deleted', 'success');
            }
        } catch (error) {
            console.error('Error deleting prompt:', error);
            UIManager.showNotification('Error deleting prompt', 'error');
        }
    }
}
