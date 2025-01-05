export class EventManager {
    constructor() {
        this.eventsContainer = document.getElementById('events-container');
        this.clearButton = document.getElementById('clear-events');
        this.maxEvents = 100;
        this.setupListeners();
        this.initializeEventStream();
    }

    setupListeners() {
        this.clearButton?.addEventListener('click', () => {
            if (this.eventsContainer) {
                this.eventsContainer.innerHTML = '';
            }
        });
    }

    async initializeEventStream() {
        try {
            const { ipcRenderer } = require('electron');
            
            ipcRenderer.send('start-event-stream');

            ipcRenderer.on('server-event', (event, data) => {
                this.addEventToDisplay(data);
            });

        } catch (error) {
            console.error('Failed to initialize event stream:', error);
            this.addErrorEvent('Failed to connect to event stream: ' + error.message);
        }
    }

    addEventToDisplay(event) {
        // Convert enum values to strings
        const levelMap = {
            0: 'info',
            1: 'warning',
            2: 'error'
        };
    
        const typeMap = {
            0: 'UNKNOWN',
            1: 'SERVICE',
            2: 'CHAT'
        };
    
        const eventElement = document.createElement('div');
        const level = levelMap[event.level] || 'info';
        eventElement.className = `event-item ${level}`;
        
        const timestamp = new Date(event.timestamp).toLocaleTimeString();
        
        eventElement.innerHTML = `
            <div class="event-header">
                <span class="event-time">${timestamp}</span>
                <span class="event-type">${typeMap[event.type] || 'UNKNOWN'}</span>
                <span class="event-level ${level}">${level.toUpperCase()}</span>
            </div>
            <div class="event-message">${event.message}</div>
            ${event.metadata ? this.formatMetadata(event.metadata) : ''}
        `;
    
        this.eventsContainer.insertBefore(eventElement, this.eventsContainer.firstChild);
    
        // Remove old events if we exceed maxEvents
        while (this.eventsContainer.children.length > this.maxEvents) {
            this.eventsContainer.removeChild(this.eventsContainer.lastChild);
        }
    }

    formatMetadata(metadata) {
        if (!metadata || Object.keys(metadata).length === 0) return '';
        
        return `
            <div class="event-metadata">
                ${Object.entries(metadata)
                    .map(([key, value]) => `<div><span>${key}:</span> ${value}</div>`)
                    .join('')}
            </div>
        `;
    }

    addErrorEvent(message) {
        this.addEventToDisplay({
            type: 'SERVICE',
            level: 'ERROR',
            timestamp: new Date().toISOString(),
            message: message,
            metadata: {}
        });
    }
}