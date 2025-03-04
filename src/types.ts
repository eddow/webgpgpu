import type { Float16Array } from '@petamoriken/float16'
import type { Bindings, BoundTypes } from './binding'
import type { BufferReader, IBuffable, ValuedBuffable } from './buffable'
import type { CodeParts } from './code'
import type { AnyInference, CreatedInferences, Inferred } from './inference'

export type InputXD<Element, SizesSpec extends readonly any[]> =
	| (SizesSpec extends [any, ...infer Rest] ? ArrayLike<InputXD<Element, Rest>> : Element)
	| ArrayBuffer // TODO: | ArrayBufferView -> { buffer: ArrayBuffer, offset+size }
export type Input0D<Element> = InputXD<Element, []>
export type Input1D<Element> = InputXD<Element, [number]>
export type Input2D<Element> = InputXD<Element, [number, number]>
export type Input3D<Element> = InputXD<Element, [number, number, number]>
export type Input4D<Element> = InputXD<Element, [number, number, number, number]>
export type AnyInput<Element = any> =
	| Input0D<Element>
	| Input1D<Element>
	| Input2D<Element>
	| Input3D<Element>
	| Input4D<Element>

export type NumericSizesSpec<SizesSpec extends readonly any[]> = {
	[K in keyof SizesSpec]: number
}

export type TypedArrayConstructor<ArrayBufferLike> = {
	new (content: number[] | number): ArrayBufferLike
	new (ab: ArrayBuffer): ArrayBufferLike
	BYTES_PER_ELEMENT: number
}
export type TypedArray = Float32Array | Float16Array | Uint32Array | Int32Array

export class WebGpGpuError extends Error {}

export class InferenceValidationError extends WebGpGpuError {
	name = 'InferenceValidationError'
}

export class ParameterError extends WebGpGpuError {
	name = 'ParameterError'
}

export class CompilationError extends WebGpGpuError {
	name = 'CompilationError'
	constructor(public messages: readonly GPUCompilationMessage[]) {
		super('Compilation error', { cause: messages })
	}
}

export class CircularImportError extends WebGpGpuError {
	name = 'CircularImportError'
	constructor(public imports: readonly PropertyKey[]) {
		super(`Circular import detected at ${imports.map(String).join(', ')}`, { cause: imports })
	}
}

export type InputType<T extends IBuffable> = //Parameters<T['value']>[0]
	T extends IBuffable<AnyInference, infer Element, infer SizesSpec>
		? InputXD<Element, SizesSpec>
		: never
export type OutputType<T extends IBuffable> = ReturnType<T['readArrayBuffer']>

export interface Kernel<
	Inferences extends AnyInference,
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
> {
	(inputs: Inputs, defaultInfers?: Partial<Record<keyof Inferences, number>>): Promise<Outputs>
	inferred: Inferences
}

// #region Kill me when bind has multiple arguments

export type WebGpGpuTypes<WGG> = WGG extends IWebGpGpu<
	infer Inferences,
	infer Inputs,
	infer Outputs
>
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
	IWebGpGpu<TypesDef['inferences'], TypesDef['inputs'], TypesDef['outputs']>

// #endregion

export interface IWebGpGpu<
	Inferences extends AnyInference,
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
> {
	// Instance properties
	readonly inferences: Inferences
	readonly device: GPUDevice
	readonly disposed: boolean
	readonly f16: boolean

	// Instance methods
	dispose(): void

	define(...definitions: CodeParts[]): IWebGpGpu<Inferences, Inputs, Outputs>

	import(...imports: PropertyKey[]): IWebGpGpu<Inferences, Inputs, Outputs>

	common<Specs extends Record<string, ValuedBuffable<Inferences>>>(
		commons: Specs
	): IWebGpGpu<Inferences, Inputs, Outputs>

	input<Specs extends Record<string, IBuffable<Inferences>>>(
		inputs: Specs
	): IWebGpGpu<Inferences, Inputs & { [K in keyof Specs]: InputType<Specs[K]> }, Outputs>

	output<Specs extends Record<string, IBuffable<Inferences>>>(
		outputs: Specs
	): IWebGpGpu<Inferences, Inputs, Outputs & { [K in keyof Specs]: OutputType<Specs[K]> }>

	workGroup(
		...size: [] | [number] | [number, number] | [number, number, number]
	): IWebGpGpu<Inferences, Inputs, Outputs>

	bind<BG extends Bindings<Inferences>>(
		group: BG
	): IWebGpGpu<
		Inferences & BoundTypes<BG>['inferences'],
		Inputs & BoundTypes<BG>['inputs'],
		Outputs & BoundTypes<BG>['outputs']
	>

	infer<
		Input extends Record<
			string,
			| Inferred
			| readonly [Inferred, Inferred]
			| readonly [Inferred, Inferred, Inferred]
			| readonly [Inferred, Inferred, Inferred, Inferred]
		>,
	>(input: Input): IWebGpGpu<Inferences & CreatedInferences<Input>, Inputs, Outputs>

	specifyInference(
		values: Partial<Inferences>,
		reason?: string
	): IWebGpGpu<Inferences, Inputs, Outputs>
}
