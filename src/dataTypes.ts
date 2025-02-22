import { Float16Array } from '@petamoriken/float16'
import { BufferReader, elementsToTypedArray } from './buffers'
import {
	type NumericSizesSpec,
	type SizeSpec,
	type WorkSizeInfer,
	resolvedSize,
} from './typedArrays'
import type { InputXD, TypedArray } from './types'

export type ValuedBuffable<
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	SizesSpec extends SizeSpec[] = SizeSpec[],
	InputSizesSpec extends SizeSpec[] = [],
	InputSpec extends number[] = number[],
> = {
	buffable: Buffable<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
	value: InputXD<OriginElement, InputSpec, Buffer>
}

export type TypedArrayConstructor<TArray extends TypedArray> = {
	new (content: number[] | number): TArray
	new (ab: ArrayBuffer): TArray
	BYTES_PER_ELEMENT: number
}

export function isBuffable(buffable: any): buffable is Buffable {
	return buffable instanceof GpGpuData
}
/**
 * Interface is used for type inference
 */
export interface Buffable<
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	SizesSpec extends SizeSpec[] = SizeSpec[],
	InputSizesSpec extends SizeSpec[] = [],
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
	toTypedArray(
		workSizeInfer: WorkSizeInfer,
		data: InputXD<OriginElement, InputSpec, Buffer>,
		actionInfo: string
	): TypedArray
	value(
		v: InputXD<OriginElement, InputSpec, Buffer>
	): ValuedBuffable<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
	readonly wgslSpecification: string
	readonly elementSize: number
	readonly transformSize: InputSizesSpec
	readTypedArray(
		buffer: Buffer,
		workSizeInferred: WorkSizeInfer
	): BufferReader<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
}

export type InputType<T extends Buffable> = Parameters<T['value']>[0]
export type OutputType<T extends Buffable> = ReturnType<T['readTypedArray']>

class GpGpuData<
	Buffer extends TypedArray,
	OriginElement,
	SizesSpec extends SizeSpec[],
	InputSizesSpec extends SizeSpec[],
	InputSpec extends number[],
> implements Buffable<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
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
		return new GpGpuData<Buffer, NewOriginElement, SizesSpec, SizesSpec, []>(
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
	toTypedArray(
		workSizeInfer: WorkSizeInfer,
		data: InputXD<OriginElement, InputSpec, Buffer>,
		required?: string
	): Buffer {
		return elementsToTypedArray<Buffer, OriginElement, InputSizesSpec>(
			this,
			workSizeInfer,
			data,
			this.size,
			required
		)
	}
	readTypedArray(
		buffer: Buffer,
		workSizeInferred: WorkSizeInfer
	): BufferReader<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec> {
		return new BufferReader<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>(
			this,
			buffer,
			resolvedSize(this.size, workSizeInferred)
		)
	}
	array<SubSizesSpec extends SizeSpec[]>(...size: SubSizesSpec) {
		return new GpGpuData<
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
	): ValuedBuffable<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec> {
		return {
			buffable: this,
			value: v,
		}
	}
}

export class GpGpuXFloat32<OriginElement> extends GpGpuData<
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

export class GpGpuXUint32<OriginElement> extends GpGpuData<Uint32Array, OriginElement, [], [], []> {
	constructor(
		elementSize: number,
		wgslSpecification: string,
		elementConvert?: (element: OriginElement) => ArrayLike<number>,
		elementRecover?: (element: ArrayLike<number>) => OriginElement
	) {
		super(Uint32Array, elementSize, wgslSpecification, [], [], elementConvert, elementRecover)
	}
}

export class GpGpuXInt32<OriginElement> extends GpGpuData<Int32Array, OriginElement, [], [], []> {
	constructor(
		elementSize: number,
		wgslSpecification: string,
		elementConvert?: (element: OriginElement) => ArrayLike<number>,
		elementRecover?: (element: ArrayLike<number>) => OriginElement
	) {
		super(Int32Array, elementSize, wgslSpecification, [], [], elementConvert, elementRecover)
	}
}
export type GpGpuSingleton<Element> = GpGpuData<TypedArray, Element, [], [], []>
