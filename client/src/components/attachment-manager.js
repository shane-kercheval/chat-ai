import { EventEmitter } from '../utils/event-emitter.js';
import { UIManager } from '../utils/ui-utils.js';
import { AttachmentStore } from '../stores/attachment-store.js';

export class AttachmentManager extends EventEmitter {

    static handlersInitialized = false;
    static searchProviders = new Set();
    static registerSearchProvider(provider) {
        AttachmentManager.searchProviders.add(provider);
    }

    constructor() {
        super();
        this.contextStrategy = 'Auto';
        this.webSearchEnabled = false;
        this.elements = {
            searchBox: document.getElementById('search-box'),
            searchResults: document.getElementById('search-results'),
            attachedItems: document.getElementById('attached-items'),
            addButton: document.querySelector('.add-attachment-button'),
            clearButton: document.querySelector('.clear-attachments-button')
        };
        // Only set up the shared handlers once (this constructor is called for each subclass)
        if (!AttachmentManager.handlersInitialized) {
            this.setupSharedHandlers();
            this.setupKeyboardShortcuts();
            AttachmentManager.handlersInitialized = true;
        }
        this.setupClearHandler();
        this.setupStrategyHandler();
        this.setupMCPToolHandler();
        AttachmentManager.registerSearchProvider(this);
    }

    setupSharedHandlers() {
        const searchContainer = document.getElementById('search-container');
        
        if (this.elements.addButton && searchContainer) {
            this.elements.addButton.addEventListener('click', (e) => {
                e.stopPropagation();
                searchContainer.style.display = 'block';
                if (this.elements.searchBox) {
                    this.elements.searchBox.value = '';
                    this.elements.searchBox.focus();
                    this.handleSearch();
                }
            });
        }

        // Handle clicks outside search area
        document.addEventListener('click', (e) => {
            if (!this.elements.searchBox?.contains(e.target) && 
                !this.elements.searchResults?.contains(e.target) &&
                !this.elements.addButton?.contains(e.target)) {
                searchContainer.style.display = 'none';
            }
        });

        // Handle escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                searchContainer.style.display = 'none';
                if (this.elements.searchBox) {
                    this.elements.searchBox.value = '';
                }
            }
        });

        if (this.elements.searchBox) {
            this.elements.searchBox.addEventListener('input', () => {
                this.handleSearch();
            });
        }
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Command+Shift+A on Mac or Ctrl+Shift+A on Windows/Linux
            if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'a') {
                e.preventDefault();  // Prevent default browser behavior
                const searchContainer = document.getElementById('search-container');
                searchContainer.style.display = 'block';
                if (this.elements.searchBox) {
                    this.elements.searchBox.value = '';
                    this.elements.searchBox.focus();
                    this.handleSearch();
                }
            }
        });
    }

    setupClearHandler() {
        if (this.elements.clearButton) {
            this.elements.clearButton.addEventListener('click', () => {
                if (this.elements.attachedItems) {
                    this.clearAttachments();
                }
            });
        }
    }

    clearAttachments() {
        if (this.elements.attachedItems) {
            this.elements.attachedItems.innerHTML = '';
            this.emit('itemsCleared');
            this.emit('attachmentsChanged'); // Add this if you need to notify about attachment changes
        }
    }

    setupStrategyHandler() {
        const strategyButton = document.querySelector('.context-strategy-button');
        strategyButton.dataset.strategy = this.contextStrategy;
        if (strategyButton) {
            strategyButton.addEventListener('click', () => {
                // Cycle through strategies
                const strategies = ['Auto', 'RAG', 'Full Text'];
                const currentIndex = strategies.indexOf(this.contextStrategy);
                this.contextStrategy = strategies[(currentIndex + 1) % strategies.length];
                // Update button state
                strategyButton.dataset.strategy = this.contextStrategy;
                // Update tooltip text
                const tooltip = strategyButton.querySelector('.icon-tooltip');
                if (tooltip) {
                    tooltip.textContent = `Context Strategy: ${this.contextStrategy}`;
                }
                AttachmentStore.getInstance().setContextStrategy(this.contextStrategy);
            });
        }
    }

    setupMCPToolHandler() {
        const toolButton = document.querySelector('.mcp-tool-button');
        if (toolButton) {
            toolButton.addEventListener('click', () => {
                this.webSearchEnabled = !this.webSearchEnabled;
                toolButton.classList.toggle('active', this.webSearchEnabled);
                AttachmentStore.getInstance().setToolsEnabled(this.webSearchEnabled);
            });
        }
    }
    
    getToolsEnabled() {
        return this.webSearchEnabled;
    }
    
    handleSearch() {
        try {
            const query = this.elements.searchBox.value.toLowerCase().trim();
            this.elements.searchResults.innerHTML = '';
    
            // Get results from all providers, no query filtering
            let allResults = [];
            for (const provider of AttachmentManager.searchProviders) {
                // Pass empty string to always get all results
                const results = provider.getSearchResults('');
                if (results && results.length > 0) {
                    // Filter results here if there is a query
                    const filteredResults = query 
                        ? results.filter(item => 
                            item.textContent.toLowerCase().includes(query)
                          )
                        : results;
                    allResults = allResults.concat(filteredResults);
                }
            }
            if (allResults.length > 0) {
                allResults.sort((a, b) => {
                    const typeOrder = {
                        'model-config': 0,
                        'context': 1,
                        'instruction': 2,
                    };
                    const typeA = a.dataset.type;
                    const typeB = b.dataset.type;
                    return (typeOrder[typeA] ?? 999) - (typeOrder[typeB] ?? 999);
                });
    
                allResults.forEach(item => {
                    this.elements.searchResults.appendChild(item);
                });
            }
        } catch (error) {
            console.error('Error handling search:', error);
            UIManager.showNotification('Error searching items', 'error');
        }
    }

    // Remove CopyhandleSearch and getSearchResults methods
    // Add this method to trigger search refresh
    refreshSearch() {
        if (this.elements.searchBox?.value) {
            this.handleSearch();
        }
    }

    isItemAttached(id) {
        return !!this.elements.attachedItems?.querySelector(
            `.attachment-item[data-id="${id}"]`
        );
    }

    createSearchResultItem(item, type, typeLetter) {
        const div = document.createElement('div');
        div.className = 'attachment-item';
        div.dataset.id = item.id;
        div.dataset.type = type;
        div.dataset.typeLetter = typeLetter;

        const nameSpan = document.createElement('span');
        nameSpan.textContent = item.name;
        div.appendChild(nameSpan);

        return div;
    }

    createAttachmentItem(item, type, typeLetter) {
        const div = document.createElement('div');
        div.className = 'attachment-item';
        div.dataset.id = item.id;
        div.dataset.type = type;
        div.dataset.typeLetter = typeLetter;

        const nameSpan = document.createElement('span');
        nameSpan.textContent = item.name;
        div.appendChild(nameSpan);

        const removeButton = document.createElement('button');
        removeButton.className = 'remove-file';
        removeButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M15 9l-6 6M9 9l6 6"/>
            </svg>
        `;

        div.appendChild(removeButton);
        return div;
    }

    clearAttachments() {
        if (this.elements.attachedItems) {
            this.elements.attachedItems.innerHTML = '';
            this.emit('itemsCleared');
        }
    }

    createUseButton(item, attachFunction) {
        const useButton = document.createElement('button');
        useButton.className = 'use-button';
        useButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
        `;
        
        const handleClick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            useButton.removeEventListener('click', handleClick);
            attachFunction(item);
            setTimeout(() => {
                useButton.addEventListener('click', handleClick);
            }, 100);
        };
    
        useButton.addEventListener('click', handleClick);
        return useButton;
    }
}
