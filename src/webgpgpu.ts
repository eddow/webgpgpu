import { activateF16 } from './atomicTypesList'
import type { BufferReader } from './buffers'
import { WgslCodeGenerator } from './code'
import {
	type Buffable,
	type InputType,
	type OutputType,
	type ValuedBuffable,
	isBuffable,
} from './dataTypes'
import { type Inferred, basicInference, infer, resolvedSize } from './inference'
import { callKernel } from './kernel/call'
import { inputGroupEntry } from './kernel/io'
import { kernelScope } from './kernel/scope'
import { type Log, log } from './log'
import { explicitWorkSize } from './typedArrays'
import { type AnyInput, ParameterError, WebGpGpuError, type WorkSize } from './types'

export interface BoundDataEntry {
	name: string
	type: Buffable
	resource: GPUBindingResource
}

/**
 * Contains the information shared in a WebGpGpu tree (root and descendants) and referencing the device
 */
interface RootInfo {
	dispose?(): void
	device?: GPUDevice
	reservedBindGroupLayout?: GPUBindGroupLayout
}

export class WebGpGpu<
	Inputs extends Record<string, AnyInput> = {},
	Outputs extends Record<string, BufferReader> = {},
	Inferences extends Record<string, Inferred> = typeof basicInference,
> extends WgslCodeGenerator {
	// #region Creation

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
					importUsage: [],
					inferences: basicInference,
					inferred: { threads: 3 },
					inferenceReasons: {},
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
		if (this.rootInfo.device) {
			this.rootInfo.dispose?.()
			this.rootInfo.device.destroy()
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
	//@Sealed
	/**
	 * Gives the inferred work size (x, y, z)
	 */
	public readonly inferences: Inferences
	private readonly inferred: Record<string, number> //var name => dimension
	private readonly inferenceReasons: Record<string, string>
	private readonly commonData: readonly BoundDataEntry[]
	private readonly inputs: Record<string, Buffable>
	private readonly outputs: Record<string, Buffable>
	private readonly workGroupSize: [number, number, number] | null
	private readonly usedNames: Set<string>
	private readonly rootInfo: RootInfo
	/**
	 * Allows hooking the library's log messages
	 */
	public static readonly log: Log = log
	private constructor(
		parent: WebGpGpu<any, any, any> | undefined,
		{
			definitions,
			importUsage,
			inferences,
			inferred,
			inferenceReasons,
			commonData,
			inputs,
			outputs,
			workGroupSize,
			usedNames,
		}: Partial<{
			definitions: string[]
			importUsage: Iterable<PropertyKey>
			inferences: Inferences
			inferred: Record<string, number>
			inferenceReasons: Record<string, string>
			commonData: BoundDataEntry[]
			inputs: Record<string, Buffable>
			outputs: Record<string, Buffable>
			workGroupSize: [number, number, number] | null
			usedNames: Iterable<string>
		}>,
		rootInfo?: RootInfo
	) {
		super(definitions ?? parent!.definitions, importUsage ?? parent!.importUsage)
		this.inferences = inferences ?? parent!.inferences
		this.inferred = inferred ?? parent!.inferred
		this.inferenceReasons = inferenceReasons ?? parent!.inferenceReasons
		this.commonData = commonData ?? parent!.commonData
		this.inputs = inputs ?? parent!.inputs
		this.outputs = outputs ?? parent!.outputs
		this.workGroupSize = workGroupSize !== undefined ? workGroupSize : parent!.workGroupSize
		this.usedNames = usedNames ? new Set(usedNames) : parent!.usedNames
		this.rootInfo = rootInfo ?? parent!.rootInfo
	}
	private checkNameConflicts(...names: string[]) {
		const conflicts = names.filter((name) => this.usedNames.has(name))
		if (conflicts.length)
			throw new ParameterError(`Parameter name conflict: ${conflicts.join(', ')}`)
		return new Set([...this.usedNames, ...names])
	}

	// #endregion Creation
	// #region Chainable

	/**
	 * Adds a definition (pre-function code) to the kernel
	 * @param definitions WGSL code to add in the end-kernel
	 * @returns Chainable
	 */
	define(...definitions: string[]) {
		return new WebGpGpu<Inputs, Outputs, Inferences>(this, {
			definitions: [...this.definitions, ...definitions],
		})
	}

	/**
	 * Adds an import usage to the kernel if not yet added
	 * @param imports Names of imports to add, must be `keyof WebGpGpu.imports`
	 * @returns Chainable
	 */
	use(...imports: PropertyKey[]) {
		const missing = imports.filter((name) => !(name in WebGpGpu.imports))
		if (missing.length) throw new ParameterError(`Unknown import: ${missing.join(', ')}`)
		const newImports = imports.filter((name) => !(name in this.importUsage))
		if (!newImports.length) return this
		return new WebGpGpu<Inputs, Outputs, Inferences>(this, {
			importUsage: [...this.importUsage, ...newImports],
		})
	}

	/**
	 * Definitions of standard imports - these can directly be edited by setting/deleting keys
	 */
	public static readonly imports: Record<PropertyKey, string> = Object.create(null)
	protected getImport(name: PropertyKey): string {
		return WebGpGpu.imports[name]
	}

	/**
	 * Adds common values to the kernel (given to the kernel but not expected as inputs)
	 * @param commons
	 * @returns Chainable
	 */
	common<Specs extends Record<string, ValuedBuffable>>(
		commons: Specs
	): WebGpGpu<Inputs, Outputs, Inferences> {
		const usedNames = this.checkNameConflicts(...Object.keys(commons))
		const { device } = this
		const inferences = { ...this.inferences }
		const inferenceReasons = { ...this.inferenceReasons }
		const newCommons = [...this.commonData]
		for (const [name, { buffable, value }] of Object.entries(commons)) {
			if (!isBuffable(buffable) || !value)
				throw new ParameterError(`Bad parameter for common \`${name}\``)
			const typedArray = buffable.toTypedArray(
				inferences,
				value,
				`common \`${name}\``,
				inferenceReasons
			)
			newCommons.push({
				name,
				type: buffable,
				resource: inputGroupEntry(
					device,
					name,
					resolvedSize(buffable.size, inferences),
					typedArray
				),
			})
		}

		return new WebGpGpu(this, {
			inferences,
			inferenceReasons,
			commonData: newCommons,
			usedNames,
		})
	}
	/**
	 * Defines kernel' inputs (with their default value if it's valued)
	 * @param inputs
	 * @returns Chainable
	 */
	input<Specs extends Record<string, Buffable>>(
		inputs: Specs
	): WebGpGpu<Inputs & Record<keyof Specs, InputType<Specs[keyof Specs]>>, Outputs, Inferences> {
		for (const [name, buffable] of Object.entries(inputs))
			if (!isBuffable(buffable)) throw new ParameterError(`Bad value for input \`${name}\``)
		return new WebGpGpu(this, {
			inputs: { ...this.inputs, ...inputs },
			usedNames: this.checkNameConflicts(...Object.keys(inputs)),
		})
	}
	/**
	 * Defines kernel' outputs
	 * @param inputs
	 * @returns Chainable
	 */
	output<Specs extends Record<string, Buffable>>(
		outputs: Specs
	): WebGpGpu<Inputs, Outputs & Record<keyof Specs, OutputType<Specs[keyof Specs]>>, Inferences> {
		return new WebGpGpu(this, {
			outputs: { ...this.outputs, ...outputs },
			usedNames: this.checkNameConflicts(...Object.keys(outputs)),
		})
	}
	/**
	 * Specifies the work group size
	 * @param size
	 * @returns
	 */
	workGroup(...size: WorkSize) {
		return new WebGpGpu<Inputs, Outputs, Inferences>(this, {
			workGroupSize: size.length ? explicitWorkSize(size) : null,
		})
	}

	/**
	 * Add a new inference variable / add a new inference value to an existing variable
	 * @param workSize
	 * @returns
	 */
	infer<Infer extends Record<string, Inferred | readonly Inferred[]>>(values: Infer) {
		const inferred = { ...this.inferred }
		const addedNames: string[] = []
		for (const [name, value] of Object.entries(values)) {
			if (name.includes('.')) throw new ParameterError(`Invalid infer name \`${name}\``)
			const d = Array.isArray(value) ? value.length : 1
			if (!(name in inferred)) {
				inferred[name] = d

				addedNames.push(name)
			} else if (inferred[name] !== d)
				throw new ParameterError(`Inference dimension conflict for \`${name}\``)
		}
		const usedNames = this.checkNameConflicts(...addedNames)
		const inferences = infer(this.inferences, values, '.infer() explicit call')
		return new WebGpGpu<Inputs, Outputs, typeof inferences>(this, {
			inferences,
			inferred,
			usedNames,
		})
	}

	// #endregion Chainable

	/**
	 *
	 * @param compute Create a kernel
	 * @param kernelWorkInf Default values to give to work-size axis if none were specified
	 * @returns
	 */
	kernel(compute: string, kernelDefaults: Partial<Record<keyof Inferences, number>> = {}) {
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
			inferences,
			inferred,
			workGroupSize,
			definitions,
			reservedBindGroupLayout,
		} = this
		const scope = guarded(() =>
			kernelScope(compute, kernelDefaults, {
				device,
				commonData,
				inputs,
				outputs,
				inferences,
				inferred,
				workGroupSize,
				definitions,
				reservedBindGroupLayout,
			})
		)
		const getDevice = () => this.device
		return Object.assign(
			// Kernel function signature
			async (inputs: Inputs, defaultInfers: Partial<Record<keyof Inferences, number>> = {}) =>
				guarded(() => callKernel(getDevice(), inputs, defaultInfers, scope)),
			{ toString: () => scope.code }
		)
	}
}
