import { Float16Array } from '@petamoriken/float16'
import { BufferReader, elementsToTypedArray } from './buffers'
import { type AnyInference, type Inferred, type SizeSpec, resolvedSize } from './inference'
import type { NumericSizesSpec } from './typedArrays'
import type { InputXD, TypedArray, TypedArrayConstructor } from './types'
type ValidateSizeSpec<
	Inferences extends Record<string, Inferred>,
	SizesSpec,
> = SizesSpec extends SizeSpec<Inferences>[] ? unknown : never
export type ValuedBuffable<
	Inferences extends Record<string, Inferred> = Record<string, Inferred>,
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	InputSizesSpec extends SizeSpec<Inferences>[] = [],
	InputSpec extends number[] = number[],
> = {
	buffable: Buffable<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
	value: InputXD<OriginElement, InputSpec, Buffer>
} & ValidateSizeSpec<Inferences, SizesSpec>
export function isBuffable(buffable: any): buffable is Buffable {
	return buffable instanceof GpGpuData
}
/**
 * Interface is needed for type inference
 */
export interface Buffable<
	Inferences extends Record<string, Inferred> = any,
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	InputSizesSpec extends SizeSpec<Inferences>[] = [],
	InputSpec extends number[] = number[],
> {
	readonly elementConvert?: (
		element: OriginElement,
		size: NumericSizesSpec<InputSizesSpec>
	) => ArrayLike<number>
	readonly elementRecover?: (
		element: ArrayLike<number>,
		size: NumericSizesSpec<InputSizesSpec>
	) => OriginElement
	readonly size: SizesSpec
	readonly bufferType: TypedArrayConstructor<Buffer>
	toTypedArray<Inferences extends Record<string, Inferred>>(
		inferences: Inferences,
		data: InputXD<OriginElement, InputSpec, Buffer>,
		reason: string,
		reasons: Record<string, string>
	): Buffer
	value(
		v: InputXD<OriginElement, InputSpec, Buffer>
	): ValuedBuffable<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
	readonly wgslSpecification: string
	readonly elementSize: number
	readonly transformSize: InputSizesSpec
	readTypedArray(
		buffer: Buffer,
		inferences: Record<string, Inferred>
	): BufferReader<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
}

export type InputType<T extends Buffable> = Parameters<T['value']>[0]
export type OutputType<T extends Buffable> = ReturnType<T['readTypedArray']>

class GpGpuData<
	Inferences extends Record<string, Inferred>,
	Buffer extends TypedArray,
	OriginElement,
	SizesSpec extends SizeSpec<Inferences>[],
	InputSizesSpec extends SizeSpec<Inferences>[],
	InputSpec extends number[],
> implements Buffable<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
{
	constructor(
		public readonly bufferType: TypedArrayConstructor<Buffer>,
		public readonly elementSize: number,
		public readonly wgslSpecification: string,
		public readonly size: SizesSpec,
		public readonly transformSize: InputSizesSpec,
		public readonly elementConvert?: (
			element: OriginElement,
			size: NumericSizesSpec<InputSizesSpec>
		) => ArrayLike<number>,
		public readonly elementRecover?: (
			element: ArrayLike<number>,
			size: NumericSizesSpec<InputSizesSpec>
		) => OriginElement
	) {}
	transform<NewOriginElement>(
		elementConvert: (
			element: NewOriginElement,
			size: NumericSizesSpec<SizesSpec>
		) => ArrayLike<number>,
		elementRecover: (
			element: ArrayLike<number>,
			size: NumericSizesSpec<SizesSpec>
		) => NewOriginElement
	) {
		return new GpGpuData<Inferences, Buffer, NewOriginElement, SizesSpec, SizesSpec, []>(
			this.bufferType,
			this.elementSize,
			this.wgslSpecification,
			this.size,
			this.size,
			elementConvert,
			elementRecover
		)
	}
	/**
	 * Creates a `TypedArray` from `data`
	 * @param workSizeInfer Size inference data
	 * @param data Original data to convert
	 * @param required Whether to modify the size inference data and mark modifications as "required" if given (The text is actually used in exception description)
	 * @returns TypedArray
	 */
	toTypedArray<Inferences extends AnyInference>(
		inferences: Inferences,
		data: InputXD<OriginElement, InputSpec, Buffer>,
		reason: string,
		reasons: Record<string, string>
	): Buffer {
		// @ts-expect-error circular ref: SizeSpec<Inferences> is defined twice
		return elementsToTypedArray<Inferences, Buffer, OriginElement, InputSizesSpec>(
			this,
			inferences,
			data,
			this.size,
			reason,
			reasons
		)
	}
	readTypedArray(
		buffer: Buffer,
		inferences: Inferences
	): BufferReader<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec> {
		return new BufferReader<
			Inferences,
			Buffer,
			OriginElement,
			SizesSpec,
			InputSizesSpec,
			InputSpec
		>(this, buffer, resolvedSize<Inferences, SizesSpec>(this.size, inferences))
	}
	array<SubSizesSpec extends SizeSpec<Inferences>[]>(...size: SubSizesSpec) {
		return new GpGpuData<
			Inferences,
			Buffer,
			OriginElement,
			[...SizesSpec, ...SubSizesSpec],
			InputSizesSpec,
			[...InputSpec, ...NumericSizesSpec<SubSizesSpec>]
		>(
			this.bufferType,
			this.elementSize,
			this.wgslSpecification,
			[...this.size, ...size],
			this.transformSize,
			this.elementConvert,
			this.elementRecover
		)
	}
	value(
		v: InputXD<OriginElement, InputSpec, Buffer>
	): ValuedBuffable<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec> {
		return {
			buffable: this,
			value: v,
		} as ValuedBuffable<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
	}
}

export class GpGpuXFloat32<OriginElement> extends GpGpuData<
	AnyInference,
	Float32Array,
	OriginElement,
	[],
	[],
	[]
> {
	constructor(
		elementSize: number,
		wgslSpecification: string,
		elementConvert?: (element: OriginElement) => ArrayLike<number>,
		elementRecover?: (element: ArrayLike<number>) => OriginElement
	) {
		super(Float32Array, elementSize, wgslSpecification, [], [], elementConvert, elementRecover)
	}
}

export class GpGpuXFloat16<OriginElement> extends GpGpuData<
	AnyInference,
	Float16Array,
	OriginElement,
	[],
	[],
	[]
> {
	constructor(
		elementSize: number,
		wgslSpecification: string,
		elementConvert?: (element: OriginElement) => ArrayLike<number>,
		elementRecover?: (element: ArrayLike<number>) => OriginElement
	) {
		super(Float16Array, elementSize, wgslSpecification, [], [], elementConvert, elementRecover)
	}
}

export class GpGpuXUint32<OriginElement> extends GpGpuData<
	AnyInference,
	Uint32Array,
	OriginElement,
	[],
	[],
	[]
> {
	constructor(
		elementSize: number,
		wgslSpecification: string,
		elementConvert?: (element: OriginElement) => ArrayLike<number>,
		elementRecover?: (element: ArrayLike<number>) => OriginElement
	) {
		super(Uint32Array, elementSize, wgslSpecification, [], [], elementConvert, elementRecover)
	}
}

export class GpGpuXInt32<OriginElement> extends GpGpuData<
	AnyInference,
	Int32Array,
	OriginElement,
	[],
	[],
	[]
> {
	constructor(
		elementSize: number,
		wgslSpecification: string,
		elementConvert?: (element: OriginElement) => ArrayLike<number>,
		elementRecover?: (element: ArrayLike<number>) => OriginElement
	) {
		super(Int32Array, elementSize, wgslSpecification, [], [], elementConvert, elementRecover)
	}
}
export type GpGpuSingleton<Element> = GpGpuData<AnyInference, TypedArray, Element, [], [], []>
