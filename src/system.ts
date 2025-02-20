export interface Log {
	warn(message: string): void
	error(message: string): void
}

export let getGpu: (() => GPU) | undefined
export function provideGpu(gpu: () => GPU) {
	getGpu = gpu
}
export let log: Log = {
	warn(message: string) {
		console.warn(message)
	},
	error(message: string) {
		console.error(message)
	},
}
export function setLog(logging: Log) {
	log = logging
}
