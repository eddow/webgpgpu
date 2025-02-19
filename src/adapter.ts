export interface System {
	getGpu(): GPU
	dispose(): void
}
export const system: Partial<System> = {}
