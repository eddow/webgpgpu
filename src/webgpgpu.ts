import { inference } from './binding'
import type { BindingType, Bindings, BoundTypes, WgslEntry } from './binding/bindings'
import { CommonBindings } from './binding/commons'
import { InferenceBindings } from './binding/inferences'
import { InputBindings } from './binding/inputs'
import { OutputBindings } from './binding/outputs'
import { type IBuffable, type IBufferReader, type ValuedBuffable, activateF16 } from './buffable'
import { type CodeParts, WgslCodeGenerator, preprocess } from './code'
import { mapEntries } from './hacks'
import { type AnyInference, type Inferred, infer3D, specifyInferences } from './inference'
import { makeKernel } from './kernel'
import { type Log, log } from './log'
import {
	type AnyInput,
	type IWebGpGpu,
	type Kernel,
	type OutputType,
	ParameterError,
	WebGpGpuError,
	WebGpuNotSupportedError,
} from './types'
import { explicitWorkSize } from './workgroup'

export type RootWebGpGpu = WebGpGpu<
	{
		'threads.x': Inferred
		'threads.y': Inferred
		'threads.z': Inferred
	},
	{},
	{}
>

/**
 * Contains the information shared in a WebGpGpu tree (root and descendants) and referencing the device
 */
interface RootInfo {
	dispose?(): void
	device?: GPUDevice
}

export class WebGpGpu<
		Inferences extends AnyInference,
		Inputs extends Record<string, AnyInput>,
		Outputs extends Record<string, IBufferReader>,
	>
	extends WgslCodeGenerator
	implements IWebGpGpu<Inferences, Inputs, Outputs>
{
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
					wgslNames: { thread: { sizes: [] } },
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
				if (!adapter) throw new WebGpuNotSupportedError('Adapter not created')
				return adapter.requestDevice(deviceDescriptor)
			})
			.then((device) => {
				if (!device) throw new WebGpuNotSupportedError('Device not created')
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
	private readonly wgslNames: Record<string, WgslEntry<Inferences>>
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
			wgslNames,
			groups,
		}: Partial<{
			definitions: CodeParts[]
			importUsage: Iterable<PropertyKey>
			inferences: Inferences
			inferred: Record<string, 1 | 2 | 3 | 4>
			inferenceReasons: Record<string, string>
			workGroupSize: [number, number, number] | null
			wgslNames: Record<string, WgslEntry<Inferences>>
			groups: Bindings<Inferences>[]
		}>,
		rootInfo?: RootInfo
	) {
		super(definitions ?? parent!.definitions, importUsage ?? parent!.importUsage)
		this.inferences = inferences ?? parent!.inferences
		this.inferred = inferred ?? parent!.inferred
		this.inferenceReasons = inferenceReasons ?? parent!.inferenceReasons
		this.workGroupSize = workGroupSize !== undefined ? workGroupSize : parent!.workGroupSize
		this.wgslNames = wgslNames ?? parent!.wgslNames
		this.rootInfo = rootInfo ?? parent!.rootInfo
		this.groups = groups ?? parent!.groups
	}
	private checkNameConflicts(names: Record<string, WgslEntry<Inferences>>) {
		const conflicts = Object.keys(names).filter((name) => name in this.wgslNames)
		if (conflicts.length)
			throw new ParameterError(`Parameter name conflict: ${conflicts.join(', ')}`)
		return { ...this.wgslNames, ...names }
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
	} //

	/**
	 * Definitions of standard imports - these can directly be edited by setting/deleting keys
	 */
	public static readonly imports: Record<PropertyKey, CodeParts> = Object.create(null)
	public static defineImports(imports: Record<PropertyKey, CodeParts | string>) {
		Object.assign(
			WebGpGpu.imports,
			mapEntries(imports, (i) => (typeof i === 'string' ? preprocess(i) : i))
		)
	}
	protected getImport(name: PropertyKey): CodeParts {
		return WebGpGpu.imports[name]
	}

	/**
	 * Adds common values to the kernel (given to the kernel but not expected as inputs)
	 * @param commons
	 * @returns Chainable
	 */
	common<Specs extends Record<string, ValuedBuffable<Inferences>>>(commons: Specs) {
		return this.bind(new CommonBindings<Inferences, Specs>(commons))
	}
	/**
	 * Defines kernel' inputs (with their default value if it's valued)
	 * @param inputs
	 * @returns Chainable
	 */
	input<Specs extends Record<string, IBuffable<Inferences>>>(inputs: Specs) {
		return this.bind(new InputBindings<Inferences, Specs>(inputs))
	}
	/**
	 * Defines kernel' outputs
	 * @param inputs
	 * @returns Chainable
	 */
	output<Specs extends Record<string, IBuffable<Inferences>>>(
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
			wgslNames: this.checkNameConflicts(group.wgslEntries),
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
	 * Kreate
	 * @param constants Constants to pass to the kernel
	 */
	kernel(constants?: Record<string, number>): Kernel<Inferences, Inputs, Outputs>
	/**
	 * Kreate
	 * @param compute WGSL code to compile
	 * @param constants Constants to pass to the kernel
	 */
	kernel(
		compute?: string,
		constants?: Record<string, GPUPipelineConstantValue>
	): Kernel<Inferences, Inputs, Outputs>
	kernel(
		compute: string | Record<string, number> = '',
		constants?: Record<string, GPUPipelineConstantValue>
	): Kernel<Inferences, Inputs, Outputs> {
		if (typeof compute === 'object') {
			constants = compute
			compute = ''
		}
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
			computations,
			initializations,
			groups,
			inferenceReasons,
			wgslNames,
		} = this
		const { kernel, code, kernelInferences } = guarded(() =>
			makeKernel<Inferences, Inputs, Outputs>(
				compute,
				constants,
				device,
				inferences,
				workGroupSize,
				declarations,
				computations,
				initializations,
				groups,
				WebGpGpu.bindingsOrder,
				wgslNames
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
		// TODO: input/output in the kernel for type-check ?
	}
}
