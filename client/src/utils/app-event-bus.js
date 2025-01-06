import { EventEmitter } from './event-emitter.js';

export class AppEventBus extends EventEmitter {
    static instance = null;

    static getInstance() {
        if (!AppEventBus.instance) {
            AppEventBus.instance = new AppEventBus();
        }
        return AppEventBus.instance;
    }
}
