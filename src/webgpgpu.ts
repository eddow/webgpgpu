import type { BufferReader } from 'buffers'
import {
	type Buffable,
	type InputType,
	type OutputType,
	type TypedArrayConstructor,
	type ValuedBuffable,
	isBuffable,
} from './dataTypes'
import { activateF16 } from './dataTypesList'
import { type Log, log } from './log'
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

interface BindingEntryDescription {
	layoutEntry: GPUBindGroupLayoutEntry
	description: string
}

interface OutputEntryDescription {
	name: string
	resource: GPUBindingResource
	encoder(commandEncoder: GPUCommandEncoder): void
	read(): Promise<TypedArray>
}

interface BoundDataEntry {
	name: string
	type: Buffable
	resource: GPUBindingResource
}

interface RootInfo {
	dispose(): void
	device?: GPUDevice
	readonly reservedBindGroupLayout: GPUBindGroupLayout
}

export class WebGpGpu<
	Inputs extends Record<string, AnyInput> = {},
	Outputs extends Record<string, BufferReader> = {},
> {
	static createRoot(root: GPUDevice, options?: { dispose?: () => void }): WebGpGpu
	static createRoot(
		root: GPUAdapter,
		options?: { dispose?: (device: GPUDevice) => void; deviceDescriptor?: GPUDeviceDescriptor }
	): Promise<WebGpGpu>
	static createRoot(
		root: GPU,
		options?: {
			dispose?: (device: GPUDevice) => void
			deviceDescriptor?: GPUDeviceDescriptor
			adapterOptions?: GPURequestAdapterOptions
		}
	): Promise<WebGpGpu>
	static createRoot(
		root: GPU | GPUAdapter | GPUDevice,
		{
			dispose = () => {},
			adapterOptions,
			deviceDescriptor,
		}: {
			dispose?: (device: GPUDevice) => void
			adapterOptions?: GPURequestAdapterOptions
			deviceDescriptor?: GPUDeviceDescriptor
		} = {}
	): Promise<WebGpGpu> | WebGpGpu {
		function create(device: GPUDevice) {
			activateF16(device.features.has('f16'))
			return new WebGpGpu(
				undefined,
				{
					workSizeInfer: {},
					definitions: [],
					commonData: [],
					inputs: {},
					outputs: {},
					workGroupSize: null,
					usedNames: new Set(['thread', 'threads']),
				},
				{
					device,
					// Share one object among all descendants
					reservedBindGroupLayout: device.createBindGroupLayout({
						label: 'reserved-bind-group-layout',
						entries: [
							{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
						],
					}),
					dispose: () => dispose(device),
				}
			)
		}
		if (root instanceof GPUDevice) return create(root)
		const adapter =
			root instanceof GPUAdapter ? Promise.resolve(root) : root.requestAdapter(adapterOptions)
		return adapter
			.then((adapter) => {
				if (!adapter) throw new Error('Adapter not created')
				return adapter.requestDevice(deviceDescriptor)
			})
			.then((device) => {
				if (!device) throw new Error('Device not created')
				return create(device)
			})
	}
	dispose() {
		if (!this.rootInfo.device) {
			this.rootInfo.dispose()
			this.rootInfo.device = undefined
		}
	}
	get disposed() {
		return !!this.rootInfo.device
	}
	get f16() {
		// TODO: Parse code and replace immediate values if needed?
		return this.device.features.has('f16')
	}
	get device() {
		if (!this.rootInfo.device) throw new Error('WebGpGpu already disposed')
		return this.rootInfo.device
	}
	/*get adapter() {
		return this.rootInfo.adapter
	}*/
	private readonly workSizeInfer: WorkSizeInfer
	private readonly definitions: readonly string[]
	private readonly commonData: readonly BoundDataEntry[]
	private readonly inputs: Record<string, Buffable>
	private readonly outputs: Record<string, Buffable>
	private readonly workGroupSize: [number, number, number] | null
	private readonly usedNames: Set<string>
	private readonly rootInfo: RootInfo
	public static readonly log: Log = log
	private constructor(
		parent: WebGpGpu<any, any> | undefined,
		{
			workSizeInfer,
			definitions,
			commonData,
			inputs,
			outputs,
			workGroupSize,
			usedNames,
		}: Partial<{
			workSizeInfer: WorkSizeInfer
			definitions: string[]
			commonData: BoundDataEntry[]
			inputs: Record<string, Buffable>
			outputs: Record<string, Buffable>
			workGroupSize: [number, number, number] | null
			usedNames: Set<string>
		}>,
		rootInfo?: RootInfo
	) {
		this.workSizeInfer = workSizeInfer ?? parent!.workSizeInfer
		this.definitions = definitions ?? parent!.definitions
		this.commonData = commonData ?? parent!.commonData
		this.inputs = inputs ?? parent!.inputs
		this.outputs = outputs ?? parent!.outputs
		this.workGroupSize = workGroupSize !== undefined ? workGroupSize : parent!.workGroupSize
		this.usedNames = usedNames ?? parent!.usedNames
		this.rootInfo = rootInfo ?? parent!.rootInfo
	}
	checkNameConflicts(...names: string[]) {
		const conflicts = names.filter((name) => this.usedNames.has(name))
		if (conflicts.length)
			throw new ParameterError(`Parameter name conflict: ${conflicts.join(', ')}`)
		return new Set([...this.usedNames, ...names])
	}

	defined(...definitions: string[]) {
		return new WebGpGpu<Inputs, Outputs>(this, {
			definitions: [...this.definitions, ...definitions],
		})
	}
	common<Specs extends Record<string, ValuedBuffable>>(
		commons: Specs
	): WebGpGpu<Omit<Inputs, keyof Specs>, Outputs> {
		const usedNames = this.checkNameConflicts(...Object.keys(commons))
		const { device } = this
		const workSizeInfer = { ...this.workSizeInfer }
		const newCommons = [...this.commonData]
		for (const [name, { buffable, value }] of Object.entries(commons)) {
			if (!isBuffable(buffable) || !value)
				throw new ParameterError(`Bad parameter for common \`${name}\``)
			const typedArray = buffable.toTypedArray(workSizeInfer, value, `common \`${name}\``)
			newCommons.push({
				name,
				type: buffable,
				resource: dimensionalInput(
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
	): WebGpGpu<Inputs & Record<keyof Specs, InputType<Specs[keyof Specs]>>, Outputs> {
		for (const [name, buffable] of Object.entries(inputs))
			if (!isBuffable(buffable)) throw new ParameterError(`Bad value for input \`${name}\``)
		return new WebGpGpu(this, {
			inputs: { ...this.inputs, ...inputs },
			usedNames: this.checkNameConflicts(...Object.keys(inputs)),
		})
	}
	output<Specs extends Record<string, Buffable>>(
		outputs: Specs
	): WebGpGpu<Inputs, Outputs & Record<keyof Specs, OutputType<Specs[keyof Specs]>>> {
		return new WebGpGpu(this, {
			outputs: { ...this.outputs, ...outputs },
			usedNames: this.checkNameConflicts(...Object.keys(outputs)),
		})
	}
	workGroup(...size: WorkSize) {
		return new WebGpGpu<Inputs, Outputs>(this, {
			workGroupSize: size.length ? explicitWorkSize(size) : null,
		})
	}
	kernel(compute: string, kernelWorkInf: WorkSizeInfer = {}, kernelRequiredInf: RequiredAxis = '') {
		try {
			const {
				device,
				rootInfo: { reservedBindGroupLayout },
			} = this

			// #region Common

			const commonBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
			const commonBindGroupEntries: GPUBindGroupEntry[] = []
			const commonBindGroupDescription: string[] = []
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
			const commonBindGroupLayout = device.createBindGroupLayout({
				label: 'common-bind-group-layout',
				entries: commonBindGroupLayoutEntries,
			})

			// #endregion Common

			// #region Input
			const inputBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
			const inputBindGroupDescription: string[] = []
			const inputsDescription: [string, number, Buffable][] = []

			for (const [name, buffable] of Object.entries(this.inputs)) {
				const binding = inputBindGroupLayoutEntries.length
				const { layoutEntry, description } = dimensionalEntryDescription(
					name,
					buffable,
					inputBindGroupIndex,
					binding,
					true
				)
				inputBindGroupLayoutEntries.push(layoutEntry)
				inputBindGroupDescription.push(description)
				inputsDescription.push([name, binding, buffable])
			}
			const inputBindGroupLayout = device.createBindGroupLayout({
				label: 'input-bind-group-layout',
				entries: inputBindGroupLayoutEntries,
			})

			// #endregion Input

			// #region Output

			const outputBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
			const outputBindGroupDescription: string[] = []
			const outputDescription: { name: string; binding: number; buffable: Buffable }[] = []

			for (const [name, buffable] of Object.entries(this.outputs)) {
				const binding = outputBindGroupLayoutEntries.length
				const { layoutEntry, description } = dimensionalEntryDescription(
					name,
					buffable,
					outputBindGroupIndex,
					binding,
					false
				)
				outputBindGroupLayoutEntries.push(layoutEntry)
				outputBindGroupDescription.push(description)
				outputDescription.push({ name, binding, buffable })
			}
			const outputBindGroupLayout = device.createBindGroupLayout({
				label: 'output-bind-group-layout',
				entries: outputBindGroupLayoutEntries,
			})

			// #endregion Output

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
${outputBindGroupDescription.join('\n')}

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
			//Create pipeline
			const pipeline = device.createComputePipeline({
				label: 'compute-pipeline',
				layout: device.createPipelineLayout({
					bindGroupLayouts: [
						reservedBindGroupLayout,
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

			// must be lambda in scope for call in kernel: Make sure no reference to device exist beside in the root, and disposition
			//  is checked on kernel call
			const getDevice = () => this.device
			return Object.assign(
				async function executeFunction(
					inputs: Inputs,
					callWorkInf: WorkSizeInfer = {},
					callRequiredInf: RequiredAxis = ''
				): Promise<Outputs> {
					try {
						// Check for device disposition and make sure it's not referenced by a scope
						const device = getDevice()
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
						// Inference can be done here as non-compulsory inference are not compelling
						const callWorkSizeInfer = applyDefaultInfer(
							workSizeInfer,
							callWorkInf,
							callRequiredInf,
							'Kernel call'
						)

						// #region Input

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
							const resource = dimensionalInput(
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
							layout: inputBindGroupLayout,
							entries: inputBindGroupEntries,
						})

						// #endregion
						// #region Reserved bind group

						for (const c of 'xyz') callWorkSizeInfer[c as 'x' | 'y' | 'z'] ??= 1
						const explicit = [callWorkSizeInfer.x, callWorkSizeInfer.y, callWorkSizeInfer.z] as [
							number,
							number,
							number,
						]

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

						// #region Output

						const outputBindGroupEntries: GPUBindGroupEntry[] = []
						const outputBuffers: OutputEntryDescription[] = []
						for (const { name, binding, buffable } of outputDescription) {
							const OutputEntryDescription = dimensionalOutput(
								device,
								name,
								resolvedSize(buffable.size, callWorkSizeInfer),
								buffable.elementSize,
								buffable.bufferType
							)
							const { resource } = OutputEntryDescription
							outputBindGroupEntries.push({
								binding,
								resource,
							})
							outputBuffers.push(OutputEntryDescription)
						}
						const outputBindGroup = device.createBindGroup({
							label: 'output-bind-group',
							layout: outputBindGroupLayout,
							entries: outputBindGroupEntries,
						})

						// #endregion
						// Encode and dispatch work

						const commandEncoder = device.createCommandEncoder({
							label: 'webGpGpu-encoder',
						})
						const passEncoder = commandEncoder.beginComputePass()
						passEncoder.setPipeline(pipeline)
						passEncoder.setBindGroup(reservedBindGroupIndex, reservedBindGroup)
						passEncoder.setBindGroup(commonBindGroupIndex, commonBindGroup)
						passEncoder.setBindGroup(inputBindGroupIndex, inputBindGroup)
						passEncoder.setBindGroup(outputBindGroupIndex, outputBindGroup)
						passEncoder.dispatchWorkgroups(...workGroups)
						passEncoder.end()

						// Add the "copy from output-buffer to read-buffer" command
						for (const { encoder } of outputBuffers) encoder(commandEncoder)

						// Submit commands
						device.queue.submit([commandEncoder.finish()])

						const reads = await Promise.all(outputBuffers.map(({ read }) => read()))

						const result: Record<string, BufferReader> = {}
						for (let i = 0; i < outputDescription.length; i++) {
							const { name, buffable } = outputDescription[i]
							result[name] = buffable.readTypedArray(reads[i], callWorkSizeInfer)
						}

						return result as Outputs
					} catch (e) {
						log.error(`Uncaught kernel error: ${(e as Error).message ?? e}`)
						throw e
					}
				},
				{ toString: () => code }
			)
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
					buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
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

function dimensionalInput(
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

function dimensionalOutput(
	device: GPUDevice,
	name: string,
	size: number[],
	elementSize: number,
	bufferType: TypedArrayConstructor<TypedArray>
): OutputEntryDescription {
	const totalSize = elementSize * bufferType.BYTES_PER_ELEMENT * size.reduce((a, b) => a * b, 1)
	switch (size.length) {
		//case 0: impossible - throws on layout
		case 1: {
			const outputBuffer = device.createBuffer({
				label: `${name}-output-buffer`,
				size: totalSize,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, // Used in compute shader
			})
			const readBuffer = device.createBuffer({
				label: `${name}-read-buffer`,
				size: totalSize,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, // Used for CPU read back
			})
			function encoder(commandEncoder: GPUCommandEncoder) {
				commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, totalSize)
			}
			async function read() {
				await readBuffer.mapAsync(GPUMapMode.READ)
				// `getMappedRange` returns a view in the GPU memory - it has to be copied then unmapped (to free GPU memory)
				const rv = new bufferType(readBuffer.getMappedRange().slice(0))
				readBuffer.unmap()
				return rv
			}
			return { resource: { buffer: outputBuffer }, encoder, read, name }
		}
		/* TODO: Textures
		case 2:
		case 3:*/
		default:
			throw new Error('Not implemented')
	}
}
