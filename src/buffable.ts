import {
	type Buffable,
	BufferReader,
	type ElementAccessor,
	type ValuedBuffable,
	elementsToTypedArray,
} from './buffers'
import { type AnyInference, type InferencesList, type SizeSpec, resolvedSize } from './inference'
import type { NumericSizesSpec } from './typedArrays'
import type { InputXD, TypedArray, TypedArrayConstructor } from './types'

export function isBuffable(buffable: any): buffable is Buffable {
	return buffable instanceof GpGpuData
}

export class GpGpuData<
	Inferences extends {},
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
		public readonly elementAccessor: ElementAccessor<OriginElement>
	) {}
	transform<NewOriginElement>(elementAccessor: ElementAccessor<NewOriginElement>) {
		return new GpGpuData<Inferences, Buffer, NewOriginElement, SizesSpec, SizesSpec, []>(
			this.bufferType,
			this.elementSize,
			this.wgslSpecification,
			this.size,
			this.size,
			elementAccessor
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
	array<const SubSizesSpec extends SizeSpec<Inferences>[]>(...size: SubSizesSpec) {
		return new GpGpuData<
			Inferences & InferencesList<SubSizesSpec>,
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
			this.elementAccessor
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
		elementAccessor: ElementAccessor<OriginElement>
	) {
		super(Float32Array, elementSize, wgslSpecification, [], [], elementAccessor)
	}
}
export type GpGpuSingleton<Element> = GpGpuData<any, TypedArray, Element, [], [], []>
