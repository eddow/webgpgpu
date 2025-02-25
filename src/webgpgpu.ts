import { activateF16 } from './atomicTypesList'
import { inference } from './binding'
import type { BindingGroup, BoundTypes } from './binding/group'
import { type Buffable, type ValuedBuffable, isBuffable } from './buffable'
import type { BufferReader } from './buffers'
import { WgslCodeGenerator } from './code'
import {
	type AnyInference,
	basicInference,
	infer3D,
	resolvedSize,
	specifyInferences,
} from './inference'
import { callKernel } from './kernel/call'
import { customBindGroupIndex, inputGroupEntry } from './kernel/io'
import { kernelScope } from './kernel/scope'
import { type Log, log } from './log'
import { explicitWorkSize } from './typedArrays'
import { type AnyInput, ParameterError, WebGpGpuError } from './types'

export type InputType<T extends Buffable> = Parameters<T['value']>[0]
export type OutputType<T extends Buffable> = ReturnType<T['readTypedArray']>
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
}
// TODO: allocate binding group + custom bindings
export interface Kernel<
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
	Inferences extends AnyInference,
> {
	(inputs: Inputs, defaultInfers?: Partial<Record<keyof Inferences, number>>): Promise<Outputs>
	inferred: Inferences
}

export type WebGpGpuTypes<WGG> = WGG extends WebGpGpu<infer Inputs, infer Outputs, infer Inferences>
	? {
			inputs: Inputs
			outputs: Outputs
			inferences: Inferences
		}
	: never

export type MixedTypes<TDs extends { inputs: any; outputs: any; inferences: any }[]> = TDs extends [
	infer First,
	...infer Rest,
]
	? First extends { inputs: any; outputs: any; inferences: any }
		? Rest extends { inputs: any; outputs: any; inferences: any }[]
			? {
					inputs: First['inputs'] & MixedTypes<Rest>['inputs']
					outputs: First['outputs'] & MixedTypes<Rest>['outputs']
					inferences: First['inferences'] & MixedTypes<Rest>['inferences']
				}
			: First
		: never
	: { inputs: {}; outputs: {}; inferences: {} }

export type MixedWebGpGpu<TypesDef extends { inputs: any; outputs: any; inferences: any }> =
	WebGpGpu<TypesDef['inputs'], TypesDef['outputs'], TypesDef['inferences']>

export class WebGpGpu<
	Inputs extends Record<string, AnyInput> = {},
	Outputs extends Record<string, BufferReader> = {},
	Inferences extends AnyInference = typeof basicInference,
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
			const zero = new WebGpGpu(
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
					usedNames: new Set(['thread']),
					groups: [],
				},
				{
					device,
					dispose: dispose && (() => dispose(device)),
				}
			)
			return zero.bind(inference({ threads: infer3D }))
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
	//@Sealed
	/**
	 * Gives the inferred work size (x, y, z)
	 */
	public readonly inferences: Inferences
	public readonly inferred: Record<string, 1 | 2 | 3 | 4> //var name => dimension
	public readonly inferenceReasons: Record<string, string>
	private readonly commonData: readonly BoundDataEntry[]
	private readonly inputs: Record<string, Buffable<Inferences>>
	private readonly outputs: Record<string, Buffable<Inferences>>
	private readonly workGroupSize: [number, number, number] | null
	private readonly usedNames: Set<string>
	private readonly rootInfo: RootInfo
	private readonly groups: BindingGroup<Inputs, Outputs, Inferences>[]
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
			groups,
		}: Partial<{
			definitions: string[]
			importUsage: Iterable<PropertyKey>
			inferences: Inferences
			inferred: Record<string, 1 | 2 | 3 | 4>
			inferenceReasons: Record<string, string>
			commonData: BoundDataEntry[]
			inputs: Record<string, Buffable<Inferences>>
			outputs: Record<string, Buffable<Inferences>>
			workGroupSize: [number, number, number] | null
			usedNames: Iterable<string>
			groups: BindingGroup<Inputs, Outputs, Inferences>[]
		}>,
		rootInfo?: RootInfo
	) {
		super(definitions ?? parent!.definitions, importUsage ?? parent!.importUsage)
		this.inferences = inferences ?? parent!.inferences
		this.inferred = inferred ?? parent!.inferred
		this.inferenceReasons = inferenceReasons ?? parent!.inferenceReasons
		this.commonData = commonData ?? parent!.commonData
		this.inputs = inputs ?? (parent!.inputs as Record<string, Buffable<Inferences>>)
		this.outputs = outputs ?? (parent!.outputs as Record<string, Buffable<Inferences>>)
		this.workGroupSize = workGroupSize !== undefined ? workGroupSize : parent!.workGroupSize
		this.usedNames = usedNames ? new Set(usedNames) : parent!.usedNames
		this.rootInfo = rootInfo ?? parent!.rootInfo
		this.groups = groups ?? parent!.groups
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
	import(...imports: PropertyKey[]) {
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
	common<Specs extends Record<string, ValuedBuffable<Inferences>>>(
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
	output<Specs extends Record<string, Buffable<Inferences>>>(
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
	workGroup(...size: [] | [number] | [number, number] | [number, number, number]) {
		return new WebGpGpu<Inputs, Outputs, Inferences>(this, {
			workGroupSize: size.length ? explicitWorkSize(size) : null,
		})
	}

	bind<BG extends BindingGroup<any, any, any>>(
		group: BG
	): WebGpGpu<
		Inputs & BoundTypes<BG>['inputs'],
		Outputs & BoundTypes<BG>['outputs'],
		Inferences & BoundTypes<BG>['inferences']
	> {
		const inferenceReasons = { ...this.inferenceReasons }
		const rv = new WebGpGpu<
			Inputs & BoundTypes<BG>['inputs'],
			Outputs & BoundTypes<BG>['outputs'],
			Inferences & BoundTypes<BG>['inferences']
		>(this, {
			groups: [...(this.groups as any[]), group as any],
			usedNames: this.checkNameConflicts(...group.wgslNames),
			inferences: specifyInferences<Inferences & BoundTypes<BG>['inferences']>(
				this.inferences,
				group.addedInferences as Partial<Inferences & BoundTypes<BG>['inferences']>,
				'binding',
				inferenceReasons
			),
			inferenceReasons,
		})
		group.setScope(this.device, this.groups.length + customBindGroupIndex)
		return rv
	}

	specifyInference(values: Partial<Inferences>, reason = '.specifyInference() explicit call') {
		return new WebGpGpu<Inputs, Outputs, Inferences>(this, {
			inferences: specifyInferences({ ...this.inferences }, values, reason, this.inferenceReasons),
		})
	}

	// #endregion Chainable

	/**
	 *
	 * @param compute Create a kernel
	 * @param kernelDefaults Default values to the inferences
	 * @returns
	 */
	kernel(
		compute: string,
		kernelDefaults: Partial<Record<keyof Inferences, number>> = {}
	): Kernel<Inputs, Outputs, Inferences> {
		function guarded<T>(fct: () => T) {
			try {
				return fct()
			} catch (e) {
				if (!(e instanceof WebGpGpuError))
					log.error(`Uncaught kernel building error: ${(e as Error).message ?? e}`)
				throw e
			}
		}
		const { device, commonData, inputs, outputs, inferences, workGroupSize, definitions, groups } =
			this
		const scope = guarded(() =>
			kernelScope<Inferences>(compute, kernelDefaults, {
				device,
				commonData,
				inputs,
				outputs,
				inferences,
				workGroupSize,
				definitions,
				groups,
			})
		)
		//
		const getDevice = () => this.device
		return Object.assign(
			// Kernel function signature
			(inputs: Inputs, defaultInfers: Partial<Record<keyof Inferences, number>> = {}) =>
				guarded(() =>
					callKernel<Inputs, Outputs, Inferences>(getDevice(), inputs, defaultInfers, groups, scope)
				),
			{ toString: () => scope.code, inferred: scope.kernelInferences }
		)
	}
}
