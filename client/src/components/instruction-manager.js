import { UIManager } from '../utils/ui-utils.js';
import { AttachmentManager } from './attachment-manager.js';
import { AttachmentStore } from '../stores/attachment-store.js';


export class InstructionManager extends AttachmentManager {
    constructor(instructions = [], instructionsList) {
        super();
        this.instructions = instructions;
        this.attachmentStore = AttachmentStore.getInstance();
        this.elements = {
            ...this.elements,
            instructionsList: instructionsList,
        };
        
        this.initialize();
        // Set up button after a short delay to ensure DOM is ready
        setTimeout(() => this.setupAddButton(), 0);
    }

    initialize() {
        try {
            this.setupAddButton();
            this.populateInstructions();
        } catch (error) {
            console.error('Error initializing instructions:', error);
            UIManager.showNotification('Error loading instructions', 'error');
        }
    }

    setupAddButton() {
        const addButton = document.getElementById('add-instruction-button');
        if (addButton) {
            this.elements.addButton = addButton;
            addButton.addEventListener('click', () => {
                UIManager.showNotification(
                    'Not yet supported. Edit ./artifacts/instructions.yaml to add or modify instructions.',
                    'warning'
                );
            });
        } else {
            console.error('Add instruction button not found');
        }
    }

    getSearchResults(query) {
        const matchingInstructions = this.instructions.filter(instruction =>
            (query === '' || instruction.name.toLowerCase().includes(query)) &&
            !this.isItemAttached(instruction.id)
        );
        return matchingInstructions.map(instruction => {
            const div = this.createSearchResultItem(instruction, 'instruction', 'I');
            div.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.attachInstruction(instruction, false);
            };
            return div;
        });
    }

    isItemAttached(instructionId) {
        const attachedInstructions = this.attachmentStore.getAttachments('instructions');
        return attachedInstructions.some(i => i.id === instructionId);
    }

    createInstructionItem(instruction) {
        try {
            const instructionDiv = document.createElement('div');
            instructionDiv.className = 'history-item';
            instructionDiv.dataset.id = instruction.id;
        
            const textContainer = document.createElement('div');
            textContainer.className = 'history-item-content prompt-content';
        
            const name = document.createElement('div');
            name.className = 'prompt-title';
            name.textContent = instruction.name;
            name.title = instruction.name;
        
            const content = document.createElement('div');
            content.className = 'prompt-content-preview';
            content.textContent = instruction.text;
            content.title = instruction.text;
        
            textContainer.appendChild(name);
            textContainer.appendChild(content);
        
            const useButton = this.createUseButton(instruction, this.attachInstruction.bind(this));
        
            instructionDiv.appendChild(textContainer);
            instructionDiv.appendChild(useButton);
        
            return instructionDiv;
        } catch (error) {
            console.error('Error creating instruction item:', error);
            throw error;
        }
    }

    attachInstruction(instruction, showSuccess = true) {
        try {
            const attachedInstructions = this.attachmentStore.getAttachments('instructions');
            if (attachedInstructions.some(i => i.id === instruction.id)) {
                UIManager.showNotification('Instruction already attached', 'info');
                return;
            }
    
            const div = this.createAttachmentItem(instruction, 'instruction', 'I');
            const removeButton = div.querySelector('.remove-file');
            removeButton.onclick = () => this.detachInstruction(instruction.id, div);
    
            this.elements.attachedItems.appendChild(div);
            this.attachmentStore.addAttachment('instructions', {
                key: instruction.id,  // Use this as the Map key
                value: instruction
            });    
            
            if (showSuccess) {
                UIManager.showNotification('Instruction attached', 'success');
            }

            const searchResults = this.elements.searchResults;
            const searchItem = searchResults.querySelector(`.attachment-item[data-id="${instruction.id}"]`);
            if (searchItem) searchItem.remove();
        } catch (error) {
            console.error('Error attaching instruction:', error);
            UIManager.showNotification('Error attaching instruction', 'error');
        }
    }

    detachInstruction(instructionId, element) {
        try {
            element.remove();
            this.attachmentStore.removeAttachment('instructions', instructionId);
            this.refreshSearch();
        } catch (error) {
            console.error('Error detaching instruction:', error);
            UIManager.showNotification('Error removing instruction', 'error');
        }
    }

    populateInstructions() {
        try {
            this.elements.instructionsList.innerHTML = '';
            this.instructions.forEach(instruction => {
                const instructionDiv = this.createInstructionItem(instruction);
                this.elements.instructionsList.appendChild(instructionDiv);
            });
        } catch (error) {
            console.error('Error populating instructions:', error);
            UIManager.showNotification('Error displaying instructions', 'error');
        }
    }

    getAttachedInstructions() {
        const attachedElements = this.elements.attachedItems.querySelectorAll('.attachment-item[data-type="instruction"]');
        return Array.from(attachedElements).map(element => {
            const instruction = this.instructions.find(i => i.id === element.dataset.id);
            return instruction;
        }).filter(Boolean);
    }

    clearAttachments() {
        this.elements.attachedItems.innerHTML = '';
        // super.clearAttachments();
        this.attachmentStore.clearAttachments('instructions');
    }
}
