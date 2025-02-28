import {
	type AnyInference,
	type InferencesList,
	type Inferred,
	type SizeSpec,
	resolvedSize,
} from '../inference'
import type { NumericSizesSpec, TypedArray, TypedArrayConstructor } from './arrays'
import type { Buffable, InputXD, ValuedBuffable } from './buffable'
import { BufferReader, type Writer, toArrayBuffer } from './io'

export function isBuffable(buffable: any): buffable is Buffable {
	return buffable instanceof GpGpuData
}

export abstract class GpGpuData<
	Inferences extends AnyInference,
	Element,
	SizesSpec extends SizeSpec<Inferences>[],
	ElementSizeSpec extends SizeSpec<Inferences>[],
> implements Buffable<Inferences, Element, SizesSpec>
{
	constructor(public readonly size: SizesSpec) {}
	abstract readonly elementSize: ElementSizeSpec
	elementByteSize(inferences: Inferences): number {
		return resolvedSize(this.elementSize, inferences).reduce((a, b) => a * b, this.bytesPerAtomic)
	}
	abstract write(buffer: ArrayBuffer): Writer<Element>
	toArrayBuffer(
		data: InputXD<Element, SizesSpec>,
		inferences: Inferences,
		reason: string,
		reasons: Record<string, string>
	): ArrayBuffer {
		return toArrayBuffer<Inferences, Element, SizesSpec>(
			this.elementByteSize(inferences),
			data,
			this.size,
			(buffer) => this.write(buffer),
			inferences,
			reason,
			reasons
		)
	}

	abstract read(buffer: ArrayBuffer): (index: number) => Element
	readArrayBuffer(
		buffer: ArrayBuffer,
		inferences: Inferences
	): BufferReader<Element, NumericSizesSpec<SizesSpec>> {
		return new BufferReader<Element, NumericSizesSpec<SizesSpec>>(
			this.read(buffer),
			resolvedSize<Inferences, SizesSpec>(this.size, inferences)
		)
	}
	array<const SubSizesSpec extends SizeSpec<Inferences>[]>(...size: SubSizesSpec) {
		return new GpGpuArrayData<
			Inferences & Record<InferencesList<SubSizesSpec>, Inferred>,
			Element,
			[...SubSizesSpec, ...SizesSpec],
			ElementSizeSpec
		>(this, [...size, ...this.size])
	}
	value(
		v: InputXD<Element, SizesSpec>
	): ValuedBuffable<Inferences, Element, SizesSpec, ElementSizeSpec> {
		return {
			buffable: this,
			value: v,
		}
	}
	abstract readonly bytesPerAtomic: number
	abstract readonly wgslSpecification: string
}

export class GpGpuArrayData<
	Inferences extends AnyInference,
	Element,
	SizesSpec extends SizeSpec<Inferences>[],
	ElementSizeSpec extends SizeSpec<Inferences>[],
> extends GpGpuData<Inferences, Element, SizesSpec, ElementSizeSpec> {
	write(buffer: ArrayBuffer): Writer<Element> {
		return this.parent.write(buffer)
	}
	read(buffer: ArrayBuffer): (index: number) => Element {
		return this.parent.read(buffer)
	}
	get bytesPerAtomic(): number {
		return this.parent.bytesPerAtomic
	}
	get wgslSpecification() {
		return this.parent.wgslSpecification
	}
	get elementSize() {
		return this.parent.elementSize
	}
	constructor(
		protected parent: GpGpuData<Inferences, Element, any, any>,
		size: SizesSpec
	) {
		super(size)
	}
	array<const SubSizesSpec extends SizeSpec<Inferences>[]>(...size: SubSizesSpec) {
		return new GpGpuArrayData<
			Inferences & Record<InferencesList<SubSizesSpec>, Inferred>,
			Element,
			[...SubSizesSpec, ...SizesSpec],
			ElementSizeSpec
		>(this.parent, [...size, ...this.size])
	}
}

export interface AtomicAccessor<Element> {
	read(array: TypedArray, index: number): Element
	write(array: TypedArray, index: number, value: Element): void
	writeMany?(array: TypedArray, index: number, values: Element[]): void
}

export class GpGpuAtomicData<Buffer extends TypedArray, Element> extends GpGpuData<
	AnyInference,
	Element,
	[],
	[]
> {
	constructor(
		public readonly bufferType: TypedArrayConstructor<Buffer>,
		public readonly atomicSize: number,
		public readonly wgslSpecification: string,
		public readonly elementAccessor: AtomicAccessor<Element>
	) {
		super([])
	}
	transform<NewElement>(elementAccessor: AtomicAccessor<NewElement>) {
		return new GpGpuAtomicData<Buffer, NewElement>(
			this.bufferType,
			this.atomicSize,
			this.wgslSpecification,
			elementAccessor
		)
	}
	get bytesPerAtomic() {
		return this.atomicSize * this.bufferType.BYTES_PER_ELEMENT
	}
	get elementSize() {
		return [] as [] // :-D
	}
	write(buffer: ArrayBuffer) {
		const typedArray = new this.bufferType(buffer)
		const { write, writeMany } = this.elementAccessor
		return {
			write: (index, value) => write(typedArray, index * this.atomicSize, value),
			writeMany:
				writeMany && ((index, values) => writeMany(typedArray, index * this.atomicSize, values)),
		} as Writer<Element>
	}
	read(buffer: ArrayBuffer) {
		const typedArray = new this.bufferType(buffer)
		return (index: number) => this.elementAccessor.read(typedArray, index * this.atomicSize)
	}
}
export type GpGpuSingleton<Element> = GpGpuAtomicData<TypedArray, Element>
