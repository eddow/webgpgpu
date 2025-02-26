import { Float16Array } from '@petamoriken/float16'
import { type Buffable, BufferReader, type ValuedBuffable, elementsToTypedArray } from './buffers'
import { type AnyInference, type SizeSpec, resolvedSize } from './inference'
import type { NumericSizesSpec } from './typedArrays'
import type { InputXD, TypedArray, TypedArrayConstructor } from './types'

export function isBuffable(buffable: any): buffable is Buffable {
	return buffable instanceof GpGpuData
}
class GpGpuData<
	Inferences extends AnyInference,
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
	toTypedArray(
		inferences: Inferences,
		data: InputXD<OriginElement, InputSpec, Buffer>,
		reason: string,
		reasons: Record<string, string>
	): Buffer {
		return elementsToTypedArray<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec>(
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
		// @ts-expect-error old rollup make an error
		return {
			buffable: this,
			value: v,
		} // as ValuedBuffable<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
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
		// `as any` for rollup version problems
		super(
			Float16Array as any,
			elementSize,
			wgslSpecification,
			[],
			[],
			elementConvert,
			elementRecover
		)
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
