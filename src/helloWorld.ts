import { system } from './system'

export async function initWebGPU() {
	const adapter = (await system.getGpu!().requestAdapter()) as GPUAdapter
	const device = await adapter.requestDevice()

	const vertexData0 = new Float32Array([0.0, 1.0, 1.0, -1.0, -1.0, -1.0])
	const vertexData1 = new Float32Array([0.0, -1.0, 1.0, 1.0, -1.0, 0.0])
	const vertexBuffer0 = device.createBuffer({
		size: vertexData0.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	})
	device.queue.writeBuffer(vertexBuffer0, 0, vertexData0)
	const vertexBuffer1 = device.createBuffer({
		size: vertexData1.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	})
	device.queue.writeBuffer(vertexBuffer1, 0, vertexData1)

	const outputBuffer = device.createBuffer({
		size: vertexData0.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, // Used in compute shader
	})

	const readBuffer = device.createBuffer({
		size: vertexData0.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, // Used for CPU read back
	})

	const computeShaderCode = /*wgsl*/ `
struct Vertex {
	@location(0) position : vec2f;
}

@group(0) @binding(0) var<storage, read> inputBuffer0 : array<Vertex>;
@group(0) @binding(1) var<storage, read> inputBuffer1 : array<Vertex>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<Vertex>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3u) {
	let index = id.x;
	if (index < arrayLength(&inputBuffer0)) {
		outputBuffer[index].position = inputBuffer0[index].position + inputBuffer1[index].position;
	}
}
`
	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
			{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
			{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
			{ binding: 0, resource: { buffer: vertexBuffer0 } },
			{ binding: 1, resource: { buffer: vertexBuffer1 } },
			{ binding: 2, resource: { buffer: outputBuffer } },
		],
	})

	const commandEncoder = device.createCommandEncoder()
	const passEncoder = commandEncoder.beginComputePass()
	passEncoder.setPipeline(pipeline)
	passEncoder.setBindGroup(0, bindGroup)
	passEncoder.dispatchWorkgroups(Math.ceil(vertexData0.length / 2 / 64))
	passEncoder.end()

	commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, vertexData0.byteLength)

	device.queue.submit([commandEncoder.finish()])

	await readBuffer.mapAsync(GPUMapMode.READ)
	const result = new Float32Array(readBuffer.getMappedRange())
	console.log('Output Data:', Array.from(result))
	outputBuffer.unmap()
}
