export interface Log {
	warn(...message: string[]): void
	error(...message: string[]): void
	info(...message: string[]): void
}
export const log: Log = {
	warn(...message: string[]) {
		console.warn(...message)
	},
	error(...message: string[]) {
		console.error(...message)
	},
	info(...message: string[]) {
		console.info(...message)
	},
}
