import { Float16Array } from '@petamoriken/float16'
import { BufferReader, elementsToTypedArray } from './buffers'
import { type AnyInference, type Inferred, type SizeSpec, resolvedSize } from './inference'
import type { NumericSizesSpec } from './typedArrays'
import type { InputXD, TypedArray, TypedArrayConstructor } from './types'

export type ValuedBuffable<
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	Inferences extends Record<string, Inferred> = Record<string, Inferred>,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	InputSizesSpec extends SizeSpec<Inferences>[] = [],
	InputSpec extends number[] = number[],
> = {
	buffable: Buffable<Buffer, OriginElement, Inferences, SizesSpec, InputSizesSpec, InputSpec>
	value: InputXD<OriginElement, InputSpec, Buffer>
}

export function isBuffable(buffable: any): buffable is Buffable {
	return buffable instanceof GpGpuData
}
/**
 * Interface is needed for type inference
 */
export interface Buffable<
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	Inferences extends Record<string, Inferred> = any,
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
	): ValuedBuffable<Buffer, OriginElement, Inferences, SizesSpec, InputSizesSpec, InputSpec>
	readonly wgslSpecification: string
	readonly elementSize: number
	readonly transformSize: InputSizesSpec
	readTypedArray(
		buffer: Buffer,
		inferences: Record<string, Inferred>
	): BufferReader<Buffer, OriginElement, Inferences, SizesSpec, InputSizesSpec, InputSpec>
}

export type InputType<T extends Buffable> = Parameters<T['value']>[0]
export type OutputType<T extends Buffable> = ReturnType<T['readTypedArray']>

class GpGpuData<
	Buffer extends TypedArray,
	OriginElement,
	Inferences extends Record<string, Inferred>,
	SizesSpec extends SizeSpec<Inferences>[],
	InputSizesSpec extends SizeSpec<Inferences>[],
	InputSpec extends number[],
> implements Buffable<Buffer, OriginElement, Inferences, SizesSpec, InputSizesSpec, InputSpec>
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
		return new GpGpuData<Buffer, NewOriginElement, Inferences, SizesSpec, SizesSpec, []>(
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
		return elementsToTypedArray<Buffer, OriginElement, Inferences, InputSizesSpec>(
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
	): BufferReader<Buffer, OriginElement, Inferences, SizesSpec, InputSizesSpec, InputSpec> {
		return new BufferReader<
			Buffer,
			OriginElement,
			Inferences,
			SizesSpec,
			InputSizesSpec,
			InputSpec
		>(this, buffer, resolvedSize<Inferences, SizesSpec>(this.size, inferences))
	}
	array<SubSizesSpec extends SizeSpec<Inferences>[]>(...size: SubSizesSpec) {
		return new GpGpuData<
			Buffer,
			OriginElement,
			Inferences,
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
	): ValuedBuffable<Buffer, OriginElement, Inferences, SizesSpec, InputSizesSpec, InputSpec> {
		return {
			buffable: this,
			value: v,
		}
	}
}

export class GpGpuXFloat32<OriginElement> extends GpGpuData<
	Float32Array,
	OriginElement,
	AnyInference,
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
	Float16Array,
	OriginElement,
	AnyInference,
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
	Uint32Array,
	OriginElement,
	AnyInference,
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
	Int32Array,
	OriginElement,
	AnyInference,
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
export type GpGpuSingleton<Element> = GpGpuData<TypedArray, Element, AnyInference, [], [], []>
