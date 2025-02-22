export * from '.'
const dynamicCreateWebGpGpu = /^Node/.test(navigator.userAgent)
	? import('./server')
	: import('./client')

export default async function createWebGpGpu(
	adapterOptions?: GPURequestAdapterOptions,
	deviceDescriptor?: GPUDeviceDescriptor
) {
	return (await dynamicCreateWebGpGpu).default(adapterOptions, deviceDescriptor)
}
