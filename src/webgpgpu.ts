import type { Buffable, InputType, ValuedBuffable } from './dataTypes'
import { activateF16 } from './dataTypesList'
import { getGpu, log } from './system'
import {
	type AnyInput,
	type RequiredAxis,
	type TypedArray,
	type WorkSizeInfer,
	applyDefaultInfer,
	resolvedSize,
} from './typedArrays'
import { type WorkSize, explicitWorkSize, workGroupCount, workgroupSize } from './workGroup'

export class ParameterError extends Error {
	name = 'ParameterError'
}
export class CompilationError extends Error {
	name = 'CompilationError'
	constructor(public cause: readonly GPUCompilationMessage[]) {
		super('Compilation error')
	}
}
const reservedBindGroupIndex = 0
const commonBindGroupIndex = 1
const inputBindGroupIndex = 2
const outputBindGroupIndex = 3

let reservedBindGroupLayout: GPUBindGroupLayout | undefined
let root: Promise<WebGpGpu<{}> | undefined> | undefined
let disposed = false

interface BindingEntryDescription {
	layoutEntry: GPUBindGroupLayoutEntry
	description: string
}

interface BoundDataEntry {
	name: string
	type: Buffable
	resource: GPUBindingResource
}

export class WebGpGpu<Inputs extends Record<string, AnyInput> = {}> {
	/**
	 * Get the root WebGpGpu instance
	 * Can resolve to undefined when the GPU has been disposed
	 */
	static get root(): Promise<WebGpGpu<{}>> {
		async function init() {
			if (!getGpu) throw new Error('Wrongly linked library usage') // client/server index makes the link
			const adapter = (await getGpu().requestAdapter()) as GPUAdapter
			const device = (await adapter.requestDevice()) as GPUDevice
			activateF16(device.features.has('f16'))
			/*device.addEventListener('uncapturederror', function (event) {
				// Re-surface the error.
				console.error('A WebGPU error was not captured:', event.error)
			})*/
			reservedBindGroupLayout = device.createBindGroupLayout({
				label: 'reserved-bind-group-layout',
				entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
			})
			return new WebGpGpu<{}>(undefined, {
				device,
				adapter,
				workSizeInfer: {},
				definitions: [],
				commonData: [],
				inputs: {},
				workGroupSize: null,
				usedNames: new Set(['thread', 'threads']),
			})
		}
		root ??= init()
		return root as Promise<WebGpGpu<{}>>
	}
	get f16() {
		// TODO: Parse code and replace immediate values if needed?
		return this.device.features.has('f16')
	}
	static dispose() {
		root = Promise.resolve(undefined)
		reservedBindGroupLayout = undefined
		disposed = true
	}
	public readonly device: GPUDevice
	public readonly adapter: GPUAdapter
	private readonly workSizeInfer: WorkSizeInfer
	private readonly definitions: readonly string[]
	private readonly commonData: readonly BoundDataEntry[]
	private readonly inputs: Record<string, Buffable>
	private readonly workGroupSize: [number, number, number] | null
	private readonly usedNames: Set<string>
	private constructor(
		parent: WebGpGpu<any> | undefined,
		{
			device,
			adapter,
			workSizeInfer,
			definitions,
			commonData,
			inputs,
			workGroupSize,
			usedNames,
		}: Partial<{
			device: GPUDevice
			adapter: GPUAdapter
			workSizeInfer: WorkSizeInfer
			definitions: string[]
			commonData: BoundDataEntry[]
			inputs: Record<string, Buffable>
			workGroupSize: [number, number, number] | null
			usedNames: Set<string>
		}>
	) {
		this.device = device ?? parent!.device
		this.adapter = adapter ?? parent!.adapter
		this.workSizeInfer = workSizeInfer ?? parent!.workSizeInfer
		this.definitions = definitions ?? parent!.definitions
		this.commonData = commonData ?? parent!.commonData
		this.inputs = inputs ?? parent!.inputs
		this.workGroupSize = workGroupSize !== undefined ? workGroupSize : parent!.workGroupSize
		this.usedNames = usedNames ?? parent!.usedNames
	}
	defined(...definitions: string[]) {
		return new WebGpGpu<Inputs>(this, {
			definitions: [...this.definitions, ...definitions],
		})
	}
	checkNameConflicts(...names: string[]) {
		const conflicts = names.filter((name) => this.usedNames.has(name))
		if (conflicts.length)
			throw new ParameterError(`Parameter name conflict: ${conflicts.join(', ')}`)
		return new Set([...this.usedNames, ...names])
	}
	common<Specs extends Record<string, ValuedBuffable>>(
		commons: Specs
	): WebGpGpu<Omit<Inputs, keyof Specs>> {
		const usedNames = this.checkNameConflicts(...Object.keys(commons))
		const { device } = this
		const workSizeInfer = { ...this.workSizeInfer }
		const newInputs = { ...this.inputs }
		const newCommons = [...this.commonData]
		for (const [name, { buffable, value }] of Object.entries(commons)) {
			if (!buffable) throw new ParameterError(`Unknown input: ${name}`)
			delete newInputs[name]
			const typedArray = buffable.toTypedArray(workSizeInfer, value, `common \`${name}\``)
			newCommons.push({
				name,
				type: buffable,
				resource: dimensionalEntry(
					device,
					name,
					resolvedSize(buffable.size, workSizeInfer),
					typedArray
				),
			})
		}

		return new WebGpGpu(this, {
			workSizeInfer,
			commonData: newCommons,
			usedNames,
		})
	}
	input<Specs extends Record<string, Buffable>>(
		inputs: Specs
	): WebGpGpu<Inputs & Record<keyof Specs, InputType<Specs[keyof Specs]>>> {
		return new WebGpGpu(this, {
			inputs: { ...this.inputs, ...inputs },
			usedNames: this.checkNameConflicts(...Object.keys(inputs)),
		})
	}
	workGroup(...size: WorkSize) {
		return new WebGpGpu<Inputs>(this, {
			workGroupSize: size.length ? explicitWorkSize(size) : null,
		})
	}
	kernel(compute: string, kernelWorkInf: WorkSizeInfer = {}, kernelRequiredInf: RequiredAxis = '') {
		if (disposed) throw new Error('GPU has been disposed')
		try {
			const { device } = this
			const commonBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
			const commonBindGroupEntries: GPUBindGroupEntry[] = []
			const commonBindGroupDescription: string[] = []
			const inputBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
			const inputBindGroupDescription: string[] = []
			const inputsDescription: [string, number, Buffable][] = []

			for (const { name, resource, type: buffable } of this.commonData) {
				const binding = commonBindGroupLayoutEntries.length

				const { layoutEntry, description } = dimensionalEntryDescription(
					name,
					buffable,
					commonBindGroupIndex,
					binding,
					true
				)
				commonBindGroupLayoutEntries.push(layoutEntry)
				commonBindGroupDescription.push(description)
				commonBindGroupEntries.push({
					binding,
					resource,
				})
			}
			for (const [name, specification] of Object.entries(this.inputs)) {
				const binding = inputBindGroupLayoutEntries.length
				const { layoutEntry, description } = dimensionalEntryDescription(
					name,
					specification,
					inputBindGroupIndex,
					binding,
					true
				)
				inputBindGroupLayoutEntries.push(layoutEntry)
				inputBindGroupDescription.push(description)
				inputsDescription.push([name, binding, specification])
			}

			const commonBindGroupLayout = device.createBindGroupLayout({
				label: 'common-bind-group-layout',
				entries: commonBindGroupLayoutEntries,
			})
			const inputBindGroupLayout = device.createBindGroupLayout({
				label: 'input-bind-group-layout',
				entries: inputBindGroupLayoutEntries,
			})
			const workSizeInfer = applyDefaultInfer(
				this.workSizeInfer,
				kernelWorkInf,
				kernelRequiredInf,
				'Kernel definition'
			)
			const workGroupSize =
				this.workGroupSize ||
				workgroupSize([workSizeInfer.x, workSizeInfer.y, workSizeInfer.z], this.device)

			const code = /*wgsl*/ `
@group(${reservedBindGroupIndex}) @binding(0) var<uniform> threads : vec3u;
${commonBindGroupDescription.join('\n')}
${inputBindGroupDescription.join('\n')}

@group(${outputBindGroupIndex}) @binding(0) var<storage, read_write> outputBuffer : array<f32>;

${this.definitions.join('\n')}

@compute @workgroup_size(${workGroupSize.join(',') || '1'})
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
		${compute}
	}
}
`
			//Compile the shader module
			const shaderModule = device.createShaderModule({ code })
			const shaderModuleCompilationInfo = shaderModule.getCompilationInfo()

			const outputBindGroupLayout = device.createBindGroupLayout({
				label: 'output-bind-group-layout',
				entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }],
			})
			//Create pipeline
			const pipeline = device.createComputePipeline({
				label: 'compute-pipeline',
				layout: device.createPipelineLayout({
					bindGroupLayouts: [
						reservedBindGroupLayout!,
						commonBindGroupLayout,
						inputBindGroupLayout,
						outputBindGroupLayout,
					],
				}),
				compute: { module: shaderModule, entryPoint: 'main' },
			})

			const commonBindGroup = device.createBindGroup({
				label: 'common-bind-group',
				layout: pipeline.getBindGroupLayout(commonBindGroupIndex),
				entries: commonBindGroupEntries,
			})

			const rv = async function executeFunction(
				inputs: Inputs,
				callWorkInf: WorkSizeInfer = {},
				callRequiredInf: RequiredAxis = ''
			) {
				try {
					if (disposed) throw new Error('GPU has been disposed')
					const messages = (await shaderModuleCompilationInfo).messages
					if (messages.length > 0) {
						let hasError = false

						for (const msg of messages) {
							const formatted = `[${msg.lineNum}:${msg.linePos}] ${msg.message}`
							if (msg.type === 'error') {
								hasError = true
								log.error(formatted)
							} else log.warn(formatted)
						}

						if (hasError) throw new CompilationError(messages)
					}
					const callWorkSizeInfer = applyDefaultInfer(
						workSizeInfer,
						callWorkInf,
						callRequiredInf,
						'Kernel call'
					)

					const usedInputs = new Set<string>()
					const inputBindGroupEntries: GPUBindGroupEntry[] = []
					for (const [name, binding, buffable] of inputsDescription) {
						usedInputs.add(name)
						// TODO: default values
						if (!inputs[name]) throw new ParameterError(`Missing input: ${name}`)
						const typeArray = buffable.toTypedArray(
							callWorkSizeInfer,
							inputs[name]!,
							`input \`${name}\``
						)
						const resource = dimensionalEntry(
							device,
							name,
							resolvedSize(buffable.size, callWorkSizeInfer),
							typeArray
						)
						inputBindGroupEntries.push({
							binding,
							resource,
						})
					}
					const unusedInputs = Object.keys(inputs).filter((name) => !usedInputs.has(name))
					if (unusedInputs.length) log.warn(`Unused inputs: ${unusedInputs.join(', ')}`)
					const inputBindGroup = device.createBindGroup({
						label: 'input-bind-group',
						layout: pipeline.getBindGroupLayout(inputBindGroupIndex),
						entries: inputBindGroupEntries,
					})

					// #region Reserved bind group

					const explicit = [callWorkSizeInfer.x, callWorkSizeInfer.y, callWorkSizeInfer.z].map(
						(v) => v ?? 1
					) as [number, number, number]

					const workGroups = workGroupCount(explicit, workGroupSize) as [number, number, number]

					const workSizeBuffer = device.createBuffer({
						size: 16,
						usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
					})
					device.queue.writeBuffer(workSizeBuffer, 0, new Uint32Array(explicit))
					const reservedBindGroup = device.createBindGroup({
						label: 'reserved-bind-group',
						layout: reservedBindGroupLayout!,
						entries: [
							{
								binding: 0,
								resource: { buffer: workSizeBuffer },
							},
						],
					})

					// #endregion

					const outputBuffer = device.createBuffer({
						size: explicit[0]! * 4,
						usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, // Used in compute shader
					})

					const readBuffer = device.createBuffer({
						size: explicit[0]! * 4,
						usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, // Used for CPU read back
					})
					const outputBindGroup = device.createBindGroup({
						label: 'output-bind-group',
						layout: outputBindGroupLayout,
						entries: [{ binding: 0, resource: { buffer: outputBuffer } }],
					})

					// Encode and dispatch work
					const commandEncoder = device.createCommandEncoder()
					const passEncoder = commandEncoder.beginComputePass()
					passEncoder.setPipeline(pipeline)
					passEncoder.setBindGroup(reservedBindGroupIndex, reservedBindGroup)
					passEncoder.setBindGroup(commonBindGroupIndex, commonBindGroup)
					passEncoder.setBindGroup(inputBindGroupIndex, inputBindGroup)
					passEncoder.setBindGroup(outputBindGroupIndex, outputBindGroup)
					passEncoder.dispatchWorkgroups(...workGroups)
					passEncoder.end()

					commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, explicit[0]! * 4)
					// Submit commands
					device.queue.submit([commandEncoder.finish()])
					let result: Float32Array
					try {
						await readBuffer.mapAsync(GPUMapMode.READ)
						result = new Float32Array(readBuffer.getMappedRange())
					} catch (e) {
						log.error((e as Error).message ?? e)
						throw new Error('GPU error', { cause: e })
					}
					outputBuffer.unmap()
					return result
				} catch (e) {
					log.error(`Uncaught kernel error: ${(e as Error).message ?? e}`)
					throw e
				}
			}
			rv.toString = () => code
			return rv
		} catch (e) {
			log.error(`Uncaught kernel building error: ${(e as Error).message ?? e}`)
			throw e
		}
	}
}

function dimensionalEntryDescription(
	name: string,
	buffable: Buffable,
	group: number,
	binding: number,
	readOnly: boolean
): BindingEntryDescription {
	switch (buffable.size.length) {
		case 0: {
			if (!readOnly)
				throw new ParameterError('Uniforms can only be read-only (cannot be used as input)')
			return {
				layoutEntry: {
					binding,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform' },
				},
				description: `@group(${group}) @binding(${binding}) var<uniform> ${name} : ${buffable.wgslSpecification};`,
			}
		}
		case 1: {
			return {
				layoutEntry: {
					binding,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'read-only-storage' },
				},
				description: `@group(${group}) @binding(${binding}) var<storage, ${readOnly ? 'read' : 'read_write'}> ${name} : array<${buffable.wgslSpecification}>;`,
			}
		}
		/* TODO: Textures
		case 2:
		case 3:*/
		default:
			throw new Error('Not implemented')
	}
}

function dimensionalEntry(
	device: GPUDevice,
	name: string,
	size: number[],
	data: TypedArray
): GPUBindingResource {
	switch (size.length) {
		case 0: {
			const buffer = device.createBuffer({
				label: name,
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			})
			device.queue.writeBuffer(buffer, 0, data)
			return { buffer }
		}
		case 1: {
			const buffer = device.createBuffer({
				label: name,
				size: data.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			})
			device.queue.writeBuffer(buffer, 0, data)
			return { buffer }
		}
		/* TODO: Textures
		case 2:
		case 3:*/
		default:
			throw new Error('Not implemented')
	}
}
