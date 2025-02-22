// shims/browser-stdout.js
const stream = window.stream || {} // Use existing stream shim
const Writable =
	stream.Writable ||
	class Writable {
		write(chunk) {
			console.log(chunk) // Minimal output to console
		}
	}

const stdout = new Writable()

window.browserStdout = function () {
	return stdout
}

export default window.browserStdout
export function __toCommonJS(mod) {
	return mod?.__esModule ? mod : { default: mod }
}
