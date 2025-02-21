import { WebGpGpu } from '../webgpgpu'

export * from '..'

if (!navigator.gpu) throw new Error('WebGPU not supported by browser')

export default function createWebGpGpu(
	adapterOptions?: GPURequestAdapterOptions,
	deviceDescriptor?: GPUDeviceDescriptor
) {
	return WebGpGpu.createRoot(navigator.gpu, { deviceDescriptor, adapterOptions })
}
