import { create, globals } from 'webgpu'
export { create, globals }
import { WebGpGpu } from '../webgpgpu'

/**
 * Has to be called *before* accessing WebGpGpu.root !
 * @param options Options passed to `webgpu`
 * @see https://github.com/dawn-gpu/node-webgpu?tab=readme-ov-file#usage
 */
export default function createWebGpGpu(
	adapterOptions?: GPURequestAdapterOptions,
	deviceDescriptor?: GPUDeviceDescriptor,
	...options: string[]
) {
	return WebGpGpu.createRoot(create(options), { deviceDescriptor, adapterOptions })
}
Object.assign(globalThis, globals)
export * from '..'
