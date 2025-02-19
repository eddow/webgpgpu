export interface System {
	getGpu: () => GPU
}
export const system: Partial<System> = {}
