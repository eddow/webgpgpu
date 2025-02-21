import type { BufferReader } from 'buffers'
import {
	type Buffable,
	type InputType,
	type OutputType,
	type ValuedBuffable,
	isBuffable,
} from './dataTypes'
import { activateF16 } from './dataTypesList'
import { callKernel } from './kernel/call'
import { ParameterError, dimensionalInput } from './kernel/io'
import { kernelScope } from './kernel/scope'
import { type Log, log } from './log'
import {
	type AnyInput,
	type RequiredAxis,
	WebGpGpuError,
	type WorkSizeInfer,
	resolvedSize,
} from './typedArrays'
import { type WorkSize, explicitWorkSize } from './workGroup'

export interface BoundDataEntry {
	name: string
	type: Buffable
	resource: GPUBindingResource
}

interface RootInfo {
	dispose?(): void
	device?: GPUDevice
	reservedBindGroupLayout?: GPUBindGroupLayout
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
			dispose,
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
					dispose: dispose && (() => dispose(device)),
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
			this.rootInfo.dispose?.()
			this.rootInfo.device = undefined
			this.rootInfo.reservedBindGroupLayout = undefined
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
	private get reservedBindGroupLayout() {
		if (!this.rootInfo.reservedBindGroupLayout) throw new Error('WebGpGpu already disposed')
		return this.rootInfo.reservedBindGroupLayout
	}
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
		function guarded<T>(fct: () => T) {
			try {
				return fct()
			} catch (e) {
				if (!(e instanceof WebGpGpuError))
					log.error(`Uncaught kernel building error: ${(e as Error).message ?? e}`)
				throw e
			}
		}
		const {
			device,
			commonData,
			inputs,
			outputs,
			workSizeInfer,
			workGroupSize,
			definitions,
			reservedBindGroupLayout,
		} = this
		const scope = guarded(() =>
			kernelScope(compute, kernelWorkInf, kernelRequiredInf, {
				device,
				commonData,
				inputs,
				outputs,
				workSizeInfer,
				workGroupSize,
				definitions,
				reservedBindGroupLayout,
			})
		)
		const getDevice = () => this.device
		return Object.assign(
			// Kernel function signature
			async (inputs: Inputs, callWorkInf: WorkSizeInfer = {}, callRequiredInf: RequiredAxis = '') =>
				guarded(() => callKernel(getDevice(), inputs, callWorkInf, callRequiredInf, scope)),
			{ toString: () => scope.code }
		)
	}
}
