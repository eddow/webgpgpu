import { system } from './adapter'
if (!system.getGpu) throw new Error('Wrongly linked library usage') // client/server index makes the link

export async function initWebGPU() {
	const adapter = (await system.getGpu!().requestAdapter()) as GPUAdapter
	const device = await adapter.requestDevice()

	const limits = adapter.limits
	console.log(limits.maxComputeInvocationsPerWorkgroup) // Max threads per workgroup
	console.log(limits.maxComputeWorkgroupSizeX, limits.maxComputeWorkgroupSizeY, limits.maxComputeWorkgroupSizeZ)

	const vertexData = new Float32Array([0.0, 1.0, 1.0, -1.0, -1.0, -1.0])
	const vertexBuffer = device.createBuffer({
		size: vertexData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	})
	device.queue.writeBuffer(vertexBuffer, 0, vertexData)

	const outputBuffer = device.createBuffer({
		size: vertexData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, // Used in compute shader
	})

	const readBuffer = device.createBuffer({
		size: vertexData.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, // Used for CPU read back
	})

	const computeShaderCode = /*wgsl*/ `
@group(0) @binding(0) var<storage, read> inputBuffer : array<vec2f>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<vec2f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3u) {
	let index = id.x;
	if (index < arrayLength(&inputBuffer)) {
		outputBuffer[index] = inputBuffer[index] * vec2f(2.0, 2.0);
	}
}
`
	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
			{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
		],
	})

	const pipelineLayout = device.createPipelineLayout({
		bindGroupLayouts: [bindGroupLayout],
	})
	const module = device.createShaderModule({ code: computeShaderCode })
	const pipeline = device.createComputePipeline({
		layout: pipelineLayout,
		compute: { module, entryPoint: 'main' },
	})

	const bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: vertexBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	})

	const commandEncoder = device.createCommandEncoder()
	const passEncoder = commandEncoder.beginComputePass()
	passEncoder.setPipeline(pipeline)
	passEncoder.setBindGroup(0, bindGroup)
	passEncoder.dispatchWorkgroups(Math.ceil(vertexData.length / 2 / 64))
	passEncoder.end()

	commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, vertexData.byteLength)

	device.queue.submit([commandEncoder.finish()])

	await readBuffer.mapAsync(GPUMapMode.READ)
	const result = new Float32Array(readBuffer.getMappedRange())
	console.log('Output Data:', Array.from(result))
	outputBuffer.unmap()
}
