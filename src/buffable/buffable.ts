import { mapEntries } from '../hacks'
import {
	type AnyInference,
	type InferencesList,
	type Inferred,
	type SizeSpec,
	resolvedSize,
} from '../inference'
import { BufferReader, type Reader, type Writer, toArrayBuffer } from './io'
import type { InputXD, NumericSizesSpec, TypedArray, TypedArrayConstructor } from './to-sort'

export type ValuedBuffable<
	Inferences extends AnyInference = AnyInference,
	Element = any,
	SizesSpec extends readonly SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	ElementSizeSpec extends readonly SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
> = {
	buffable: Buffable<Inferences, Element, SizesSpec, ElementSizeSpec>
	value: InputXD<Element, SizesSpec>
}

export function isBuffable(buffable: any): buffable is Buffable {
	return buffable instanceof Buffable
}

export abstract class Buffable<
	Inferences extends AnyInference = any,
	Element = any,
	SizesSpec extends readonly SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	ElementSizeSpec extends readonly SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
> {
	constructor(public readonly size: SizesSpec) {}
	abstract readonly elementSize: ElementSizeSpec
	elementByteSize(inferences: Inferences): number {
		return resolvedSize(this.elementSize, inferences).reduce((a, b) => a * b, this.bytesPerAtomic)
	}
	abstract writer(buffer: ArrayBuffer): Writer<Element>
	toArrayBuffer(
		data: InputXD<Element, SizesSpec>,
		inferences: Inferences,
		reason = 'given input',
		reasons: Record<string, string> | string = 'hardcoded'
	): ArrayBuffer {
		if (typeof reasons === 'string')
			reasons = mapEntries(inferences, (i) => (i === undefined ? undefined : (reasons as string)))
		return toArrayBuffer<Inferences, Element, SizesSpec>(
			this.elementByteSize(inferences),
			data,
			this.size,
			(buffer) => this.writer(buffer),
			inferences,
			reason,
			reasons
		)
	}

	abstract reader(buffer: ArrayBuffer): Reader<Element>
	readArrayBuffer(
		buffer: ArrayBuffer,
		inferences: Inferences
	): BufferReader<Element, NumericSizesSpec<SizesSpec>> {
		return new BufferReader<Element, NumericSizesSpec<SizesSpec>>(
			this.reader(buffer),
			buffer,
			resolvedSize<Inferences, SizesSpec>(this.size, inferences)
		)
	}
	array<const SubSizesSpec extends readonly SizeSpec<Inferences>[]>(...size: SubSizesSpec) {
		if (this.size.length > 0) throw Error('Making array of nor-scalar nor array')
		return new BuffableArray<
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
	abstract get base(): Buffable<Inferences, Element, [], ElementSizeSpec>
}

export class BuffableArray<
	Inferences extends AnyInference,
	Element,
	SizesSpec extends readonly SizeSpec<Inferences>[],
	ElementSizeSpec extends readonly SizeSpec<Inferences>[],
> extends Buffable<Inferences, Element, SizesSpec, ElementSizeSpec> {
	writer(buffer: ArrayBuffer): Writer<Element> {
		return this.parent.writer(buffer)
	}
	reader(buffer: ArrayBuffer) {
		return this.parent.reader(buffer)
	}
	get bytesPerAtomic() {
		return this.parent.bytesPerAtomic
	}
	// TODO: show array here?
	get wgslSpecification() {
		return this.parent.wgslSpecification
	}
	get elementSize() {
		return this.parent.elementSize
	}
	get base() {
		return this.parent.base
	}
	constructor(
		protected parent: Buffable<Inferences, Element, any, any>,
		size: SizesSpec
	) {
		super(size)
	}
	array<const SubSizesSpec extends readonly SizeSpec<Inferences>[]>(...size: SubSizesSpec) {
		return new BuffableArray<
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
}

export class BuffableAtomic<Buffer extends TypedArray, Element> extends Buffable<
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
		return new BuffableAtomic<Buffer, NewElement>(
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
	get base() {
		return this
	}
	writer(buffer: ArrayBuffer): Writer<Element> {
		const typedArray = new this.bufferType(buffer)
		const { write } = this.elementAccessor
		return (index, value) => write(typedArray, index * this.atomicSize, value)
	}
	reader(buffer: ArrayBuffer) {
		const typedArray = new this.bufferType(buffer)
		return (index: number) => this.elementAccessor.read(typedArray, index * this.atomicSize)
	}
}
