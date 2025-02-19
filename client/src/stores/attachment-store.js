import { EventEmitter } from '../utils/event-emitter.js';

export class AttachmentStore extends EventEmitter {
    static instance = null;
    
    constructor() {
        super();
        this.attachments = {
            resources: new Map(),
            instructions: new Map(),
            contexts: new Map()
        };
        this.contextStrategy = 'Auto';
        this.toolsEnabled = false;
    }

    addAttachment(type, { key, value }) {
        if (!this.attachments[type]) {
            throw new Error(`Unknown attachment type: ${type}`);
        }
        this.attachments[type].set(key, value);
        this.emit(`${type}:added`, value);
    }

    removeAttachment(type, key) {
        if (!this.attachments[type]) {
            throw new Error(`Unknown attachment type: ${type}`);
        }
        this.attachments[type].delete(key);
        this.emit(`${type}:removed`, key);
    }

    getAttachments(type) {
        if (!this.attachments[type]) {
            throw new Error(`Unknown attachment type: ${type}`);
        }
        return Array.from(this.attachments[type].values());
    }

    clearAttachments(type) {
        if (!this.attachments[type]) {
            throw new Error(`Unknown attachment type: ${type}`);
        }
        this.attachments[type].clear();
        this.emit(`${type}:cleared`);
    }

    setContextStrategy(strategy) {
        this.contextStrategy = strategy;
    }

    getContextStrategy() {
        return this.contextStrategy;
    }

    setToolsEnabled(enabled) {
        this.toolsEnabled = enabled;
        this.emit('toolsEnabled', enabled);
    }

    getToolsEnabled() {
        return this.toolsEnabled;
    }

    static getInstance() {
        if (!AttachmentStore.instance) {
            AttachmentStore.instance = new AttachmentStore();
        }
        return AttachmentStore.instance;
    }
}
