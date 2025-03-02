import { inference } from './binding'
import type { BindingType, Bindings, BoundTypes } from './binding/bindings'
import { CommonBindings } from './binding/commons'
import { InferenceBindings } from './binding/inferences'
import { InputBindings } from './binding/inputs'
import { OutputBindings } from './binding/outputs'
import {
	type AnyInput,
	type Buffable,
	type BufferReader,
	type ValuedBuffable,
	activateF16,
} from './buffable'
import { type CodeParts, WgslCodeGenerator } from './code'
import { type AnyInference, type Inferred, infer3D, specifyInferences } from './inference'
import { kernelScope } from './kernel'
import { type Log, log } from './log'
import { ParameterError, WebGpGpuError } from './types'
import { explicitWorkSize } from './workgroup'

export type InputType<T extends Buffable> = Parameters<T['value']>[0]
export type OutputType<T extends Buffable> = ReturnType<T['readArrayBuffer']>

/**
 * Contains the information shared in a WebGpGpu tree (root and descendants) and referencing the device
 */
interface RootInfo {
	dispose?(): void
	device?: GPUDevice
}

export interface Kernel<
	Inferences extends AnyInference,
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
> {
	(inputs: Inputs, defaultInfers?: Partial<Record<keyof Inferences, number>>): Promise<Outputs>
	inferred: Inferences
}

// #region Kill me when bind has multiple arguments

export type WebGpGpuTypes<WGG> = WGG extends WebGpGpu<infer Inferences, infer Inputs, infer Outputs>
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
	WebGpGpu<TypesDef['inferences'], TypesDef['inputs'], TypesDef['outputs']>

// #endregion

export type RootWebGpGpu = WebGpGpu<
	{
		'threads.x': Inferred
		'threads.y': Inferred
		'threads.z': Inferred
	},
	{},
	{}
>

export class WebGpGpu<
	Inferences extends AnyInference,
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
> extends WgslCodeGenerator {
	/**
	 * This is the order the bindings are processed through at run time
	 * 1 inputs - usually fix the values of inferences at run-time from parameters
	 * 2 inferences - defaults values to `1` and actually provide the inferred values to the GPU
	 */
	static bindingsOrder: BindingType<any>[] = [InputBindings, InferenceBindings]
	// #region Creation

	static createRoot(device: GPUDevice, options?: { dispose?: () => void }): RootWebGpGpu
	static createRoot(
		adapter: GPUAdapter,
		options?: { dispose?: (device: GPUDevice) => void; deviceDescriptor?: GPUDeviceDescriptor }
	): Promise<RootWebGpGpu>
	static createRoot(
		gpu: GPU,
		options?: {
			dispose?: (device: GPUDevice) => void
			deviceDescriptor?: GPUDeviceDescriptor
			adapterOptions?: GPURequestAdapterOptions
		}
	): Promise<RootWebGpGpu>
	static createRoot(
		from: GPU | GPUAdapter | GPUDevice,
		{
			dispose,
			adapterOptions,
			deviceDescriptor,
		}: {
			dispose?: (device: GPUDevice) => void
			adapterOptions?: GPURequestAdapterOptions
			deviceDescriptor?: GPUDeviceDescriptor
		} = {}
	): Promise<RootWebGpGpu> | RootWebGpGpu {
		function create(device: GPUDevice) {
			activateF16(device.features.has('f16'))
			const zero = new WebGpGpu(
				undefined, // this forces all the "optional" new values to be given
				{
					importUsage: [],
					inferences: {},
					inferred: {},
					inferenceReasons: {},
					definitions: [],
					workGroupSize: null,
					usedNames: new Set(['thread']),
					groups: [],
				},
				{
					device,
					dispose: dispose && (() => dispose(device)),
				}
			)
			return zero.bind(
				inference<{}, { threads: readonly [undefined, undefined, undefined] }>({
					threads: infer3D,
				})
			)
		}
		if (from instanceof GPUDevice) return create(from)
		const adapter =
			from instanceof GPUAdapter ? Promise.resolve(from) : from.requestAdapter(adapterOptions)
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
	private readonly workGroupSize: [number, number, number] | null
	private readonly usedNames: Set<string>
	private readonly rootInfo: RootInfo
	private readonly groups: Bindings<Inferences>[]
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
			workGroupSize,
			usedNames,
			groups,
		}: Partial<{
			definitions: CodeParts[]
			importUsage: Iterable<PropertyKey>
			inferences: Inferences
			inferred: Record<string, 1 | 2 | 3 | 4>
			inferenceReasons: Record<string, string>
			workGroupSize: [number, number, number] | null
			usedNames: Iterable<string>
			groups: Bindings<Inferences>[]
		}>,
		rootInfo?: RootInfo
	) {
		super(definitions ?? parent!.definitions, importUsage ?? parent!.importUsage)
		this.inferences = inferences ?? parent!.inferences
		this.inferred = inferred ?? parent!.inferred
		this.inferenceReasons = inferenceReasons ?? parent!.inferenceReasons
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
	define(...definitions: CodeParts[]) {
		return new WebGpGpu<Inferences, Inputs, Outputs>(this, {
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
		return new WebGpGpu<Inferences, Inputs, Outputs>(this, {
			importUsage: [...this.importUsage, ...newImports],
		})
	}

	/**
	 * Definitions of standard imports - these can directly be edited by setting/deleting keys
	 */
	public static readonly imports: Record<PropertyKey, CodeParts> = Object.create(null)
	protected getImport(name: PropertyKey): CodeParts {
		return WebGpGpu.imports[name]
	}

	/**
	 * Adds common values to the kernel (given to the kernel but not expected as inputs)
	 * @param commons
	 * @returns Chainable
	 */
	common<Specs extends Record<string, ValuedBuffable<Inferences>>>(
		commons: Specs
	): WebGpGpu<Inferences, Inputs, Outputs> {
		return this.bind(new CommonBindings<Inferences, Specs>(commons))
	}
	/**
	 * Defines kernel' inputs (with their default value if it's valued)
	 * @param inputs
	 * @returns Chainable
	 */
	input<Specs extends Record<string, Buffable<Inferences>>>(inputs: Specs) {
		return this.bind(new InputBindings<Inferences, Specs>(inputs))
	}
	/**
	 * Defines kernel' outputs
	 * @param inputs
	 * @returns Chainable
	 */
	output<Specs extends Record<string, Buffable<Inferences>>>(
		outputs: Specs
	): WebGpGpu<Inferences, Inputs, Outputs & { [K in keyof Specs]: OutputType<Specs[K]> }> {
		return this.bind(new OutputBindings<Inferences, Specs>(outputs))
	}
	/**
	 * Specifies the work group size
	 * @param size
	 * @returns
	 */
	workGroup(...size: [] | [number] | [number, number] | [number, number, number]) {
		return new WebGpGpu<Inferences, Inputs, Outputs>(this, {
			workGroupSize: size.length ? explicitWorkSize(size) : null,
		})
	}

	bind<BG extends Bindings<Inferences>>(
		group: BG
	): WebGpGpu<
		Inferences & BoundTypes<BG>['inferences'],
		Inputs & BoundTypes<BG>['inputs'],
		Outputs & BoundTypes<BG>['outputs']
	> {
		const inferenceReasons = { ...this.inferenceReasons }
		const inferences = specifyInferences<Inferences & BoundTypes<BG>['inferences']>(
			{ ...this.inferences },
			group.addedInferences as Partial<Inferences & BoundTypes<BG>['inferences']>,
			'binding',
			inferenceReasons
		)
		const rv = new WebGpGpu<
			Inferences & BoundTypes<BG>['inferences'],
			Inputs & BoundTypes<BG>['inputs'],
			Outputs & BoundTypes<BG>['outputs']
		>(this, {
			groups: [...(this.groups as any[]), group as any],
			usedNames: this.checkNameConflicts(...Object.keys(group.wgslEntries)),
			inferences,
			inferenceReasons,
		})
		group.setScope(this.device, inferences, inferenceReasons)
		return rv
	}

	infer<
		Input extends Record<
			string,
			| Inferred
			| readonly [Inferred, Inferred]
			| readonly [Inferred, Inferred, Inferred]
			| readonly [Inferred, Inferred, Inferred, Inferred]
		>,
	>(input: Input) {
		return this.bind(new InferenceBindings(input))
	}

	/**
	 * Specify some inferences explicitly
	 * @param values Values to specify
	 * @param reason Reason for specifying those values - Mainly used for debugging purposes
	 * @returns
	 */
	specifyInference(values: Partial<Inferences>, reason = '.specifyInference() explicit call') {
		return new WebGpGpu<Inferences, Inputs, Outputs>(this, {
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
	): Kernel<Inferences, Inputs, Outputs> {
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
			inferences,
			workGroupSize,
			declarations,
			initializations,
			groups,
			inferenceReasons,
		} = this
		const { kernel, code, kernelInferences } = guarded(() =>
			kernelScope<Inferences, Inputs, Outputs>(
				compute,
				kernelDefaults,
				device,
				inferences,
				workGroupSize,
				declarations,
				initializations,
				groups,
				WebGpGpu.bindingsOrder
			)
		)
		//
		const getDevice = () => this.device
		return Object.assign(
			// Kernel function signature
			(inputs: Inputs, defaultInfers: Partial<Record<keyof Inferences, number>> = {}) =>
				guarded(() => kernel(getDevice(), inputs, defaultInfers, inferenceReasons)),
			{ toString: () => code, inferred: kernelInferences }
		)
	}
}
