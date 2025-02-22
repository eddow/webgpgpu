// shims/process.js
window.process = globalThis.process = {
	env: {}, // Minimal shim for Mocha/Chai
}
