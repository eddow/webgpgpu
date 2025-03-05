import { WebGpuNotSupportedError } from 'src/types'
import { WebGpGpu } from '../webgpgpu'

export * from '..'

export default function createWebGpGpu(
	adapterOptions?: GPURequestAdapterOptions,
	deviceDescriptor?: GPUDeviceDescriptor
) {
	if (!navigator.gpu)
		return Promise.reject(new WebGpuNotSupportedError('WebGPU not supported by browser'))
	return WebGpGpu.createRoot(navigator.gpu, { deviceDescriptor, adapterOptions })
}
