export interface Log {
	warn(...message: string[]): void
	error(...message: string[]): void
	info(...message: string[]): void
}
export const log: Log = {
	warn(...message: string[]) {
		// biome-ignore lint/suspicious/noConsole: <explanation>
		console.warn(...message)
	},
	error(...message: string[]) {
		// biome-ignore lint/suspicious/noConsole: <explanation>
		console.error(...message)
	},
	info(...message: string[]) {
		// biome-ignore lint/suspicious/noConsole: <explanation>
		console.info(...message)
	},
}
