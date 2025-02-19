import { system } from './adapter'
import type { GpGpuData, TypedArray } from './dataTypes'
import { type WorkSize, explicitWorkSize, workGroupCount } from './workGroup'

interface WgslTypedData {
	name: string
	wgslType: string
	array: TypedArray
	unique: boolean
}

const reservedBindGroupIndex = 0
const commonBindGroupIndex = 1
//const parametricBindGroupIndex = 2
const outputBindGroupIndex = 2

let reservedBindGroupLayout: GPUBindGroupLayout | undefined
let root: Promise<WebGpGpu | undefined> | undefined
let disposed = false
export class WebGpGpu {
	/**
	 * Get the root WebGpGpu instance
	 * Can resolve to undefined when the GPU has been disposed
	 */
	static get root(): Promise<WebGpGpu | undefined> {
		async function init() {
			const adapter = (await system.getGpu!().requestAdapter()) as GPUAdapter
			const device = (await adapter.requestDevice()) as GPUDevice
			/*device.addEventListener('uncapturederror', function (event) {
				// Re-surface the error.
				console.error('A WebGPU error was not captured:', event.error)
			})*/
			reservedBindGroupLayout = device.createBindGroupLayout({
				label: 'reserved-bind-group-layout',
				entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
			})
			return new WebGpGpu(device, adapter, [], [])
		}
		root ??= init()
		return root
	}
	static dispose() {
		root = Promise.resolve(undefined)
		reservedBindGroupLayout = undefined
		system.dispose?.()
		disposed = true
	}
	private constructor(
		public readonly device: GPUDevice,
		public readonly adapter: GPUAdapter,
		protected readonly definitions: string[],
		protected readonly commonData: WgslTypedData[]
	) {}
	defined(...definitions: string[]) {
		return new WebGpGpu(
			this.device,
			this.adapter,
			[...this.definitions, ...definitions],
			this.commonData
		)
	}
	commonArray<
		Buffer extends TypedArray,
		FlatArray extends any[],
		Element extends number | number[],
		OriginElement,
	>(
		name: string,
		type: GpGpuData<Buffer, FlatArray, Element, OriginElement>,
		value: Buffer | OriginElement[] | FlatArray,
		size: number
	) {
		return new WebGpGpu(this.device, this.adapter, this.definitions, [
			...this.commonData,
			{
				wgslType: type.wgslSpecification,
				array: type.writeArray(value, size),
				unique: false,
				name,
			},
		])
	}
	commonUniform<
		Buffer extends TypedArray,
		FlatArray extends any[],
		Element extends number | number[],
		OriginElement,
	>(
		name: string,
		type: GpGpuData<Buffer, FlatArray, Element, OriginElement>,
		value: Buffer | OriginElement | FlatArray
	) {
		return new WebGpGpu(this.device, this.adapter, this.definitions, [
			...this.commonData,
			{ wgslType: type.wgslSpecification, array: type.writeUnique(value), unique: true, name },
		])
	}
	kernel(workGroupSize: WorkSize, compute: string) {
		if (disposed) throw new Error('GPU has been disposed')
		const { device } = this
		const commonBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
		const commonBindGroupEntries: GPUBindGroupEntry[] = []
		const commonBindGroupDescription: string[] = []
		for (const { wgslType, array, unique, name } of this.commonData) {
			const buffer = device.createBuffer({
				label: name,
				size: array.byteLength,
				usage: (unique ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST,
			})
			device.queue.writeBuffer(buffer, 0, array)
			commonBindGroupEntries.push({
				binding: commonBindGroupEntries.length,
				resource: { buffer },
			})
			commonBindGroupLayoutEntries.push({
				binding: commonBindGroupLayoutEntries.length,
				visibility: GPUShaderStage.COMPUTE,
				buffer: { type: unique ? 'uniform' : 'read-only-storage' },
			})
			commonBindGroupDescription.push(
				`@group(${commonBindGroupIndex}) @binding(${commonBindGroupEntries.length - 1}) var<${
					unique ? 'uniform' : 'storage, read'
				}> ${name} : ${unique ? wgslType : `array<${wgslType}>`};`
			)
		}
		// 🔹 Create bind group layout (common data at binding 0, dependent at 1)
		const commonBindGroupLayout = device.createBindGroupLayout({
			label: 'common-bind-group-layout',
			entries: commonBindGroupLayoutEntries,
		})

		const code = /*wgsl*/ `
@group(${reservedBindGroupIndex}) @binding(0) var<uniform> threads : vec3u;

${commonBindGroupDescription.join('\n')}

@group(2) @binding(0) var<storage, read_write> outputBuffer : array<f32>;

${this.definitions.join('\n')}

@compute @workgroup_size(${workGroupSize.join(',')})
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
		${compute}
	}
}
`
		// 🔹 Compile the shader module
		const shaderModule = device.createShaderModule({ code })
		const shaderModuleCompilationInfo = shaderModule.getCompilationInfo()

		const outputBindGroupLayout = device.createBindGroupLayout({
			label: 'output-bind-group-layout',
			entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }],
		})
		// 🔹 Create pipeline
		const pipeline = device.createComputePipeline({
			label: 'compute-pipeline',
			layout: device.createPipelineLayout({
				bindGroupLayouts: [reservedBindGroupLayout!, commonBindGroupLayout, outputBindGroupLayout],
			}),
			compute: { module: shaderModule, entryPoint: 'main' },
		})

		const commonBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(commonBindGroupIndex),
			entries: commonBindGroupEntries,
		})
		const rv = async function executeFunction(workSize: WorkSize) {
			if (disposed) throw new Error('GPU has been disposed')
			const messages = (await shaderModuleCompilationInfo).messages
			if (messages.length > 0) {
				let hasError = false

				for (const msg of messages) {
					const formatted = `[${msg.lineNum}:${msg.linePos}] ${msg.message}`
					if (msg.type === 'error') {
						hasError = true
						console.error(formatted)
					} else console.warn(formatted)
				}

				if (hasError) throw new Error('Compilation error')
			}
			const explicit = explicitWorkSize(workSize)
			const workGroups = workGroupCount(explicit, workGroupSize) as [number, number, number]

			const workSizeBuffer = device.createBuffer({
				size: 12,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			})
			device.queue.writeBuffer(workSizeBuffer, 0, new Uint32Array(explicit))

			const reservedBindGroup = device.createBindGroup({
				layout: reservedBindGroupLayout!,
				entries: [
					{
						binding: 0,
						resource: { buffer: workSizeBuffer },
					},
				],
			})

			const outputBuffer = device.createBuffer({
				size: workSize[0] * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, // Used in compute shader
			})

			const readBuffer = device.createBuffer({
				size: workSize[0] * 4,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, // Used for CPU read back
			})
			const outputBindGroup = device.createBindGroup({
				layout: outputBindGroupLayout,
				entries: [{ binding: 0, resource: { buffer: outputBuffer } }],
			})
			// 🔹 Encode and dispatch work
			const commandEncoder = device.createCommandEncoder()
			const passEncoder = commandEncoder.beginComputePass()
			passEncoder.setPipeline(pipeline)
			passEncoder.setBindGroup(reservedBindGroupIndex, reservedBindGroup)
			passEncoder.setBindGroup(commonBindGroupIndex, commonBindGroup)
			passEncoder.setBindGroup(outputBindGroupIndex, outputBindGroup)
			passEncoder.dispatchWorkgroups(...workGroups)
			passEncoder.end()

			commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, workSize[0] * 4)
			// 🔹 Submit commands
			device.queue.submit([commandEncoder.finish()])
			await readBuffer.mapAsync(GPUMapMode.READ)
			const result = new Float32Array(readBuffer.getMappedRange())
			outputBuffer.unmap()
			return result
		}
		rv.toString = () => code
		return rv
	}
}
