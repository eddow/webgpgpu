import { create, globals } from 'webgpu'
export { create, globals }
import { provideGpu } from '../system'

export let gpuOptions: string[] = []
/**
 * Has to be called *before* accessing WebGpGpu.root !
 * @param options
 */
export function setWebGpuOptions(...options: string[]) {
	gpuOptions = options
}
provideGpu(() => create(gpuOptions)!)
Object.assign(globalThis, globals)
export * from '..'
