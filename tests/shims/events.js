class EventEmitter {
	constructor() {
		this._events = new Map()
	}
	on(event, listener) {
		if (!this._events.has(event)) this._events.set(event, [])
		this._events.get(event).push(listener)
	}
	emit(event, ...args) {
		if (this._events.has(event)) {
			for (const listener of this._events.get(event)) {
				listener(...args)
			}
		}
	}
	off(event, listener) {
		if (this._events.has(event)) {
			this._events.set(
				event,
				this._events.get(event).filter((l) => l !== listener)
			)
		}
	}
}
window.EventEmitter = EventEmitter // Expose globally
