// shims/stream.js
// Define a minimal Writable stream class for the browser
class Writable {
	write(chunk) {
		// Output to console as a fallback (mimics stdout-like behavior)
		console.log(typeof chunk === 'string' ? chunk : chunk.toString())
	}
}

// Optionally, add more stream classes if needed (e.g., Readable, Duplex)
// For now, Writable is enough for browser-stdout and similar uses
export const stream = {
	Writable: Writable,
}

// Set on window for browser compatibility
window.stream = stream
