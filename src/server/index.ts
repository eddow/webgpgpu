import { create as createGPU, globals } from 'webgpu'
export { createGPU }
import { WebGpGpu } from '../webgpgpu'

/**
 * @param options Options passed to `webgpu`
 * @see https://github.com/dawn-gpu/node-webgpu?tab=readme-ov-file#usage
 */
export default function createWebGpGpu(
	adapterOptions?: GPURequestAdapterOptions,
	deviceDescriptor?: GPUDeviceDescriptor,
	...options: string[]
) {
	return WebGpGpu.createRoot(createGPU(options), { deviceDescriptor, adapterOptions })
}
Object.assign(globalThis, globals)
export * from '..'
