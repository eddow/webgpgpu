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
	const gpu = createGPU(options)

	// dawn.node requires the object returned by `create()` to stay referenced
	// for the whole lifetime of the underlying implementation.
	return WebGpGpu.createRoot(gpu, {
		deviceDescriptor,
		adapterOptions,
		dispose: () => void gpu,
	})
}
Object.assign(globalThis, globals)

export * from '..'
