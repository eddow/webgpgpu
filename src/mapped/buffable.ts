import type { AnyInference, SizeSpec } from '../inference'
import type { NumericSizesSpec } from './arrays'
import type { BufferReader } from './io'

export type Input0D<Element> = Element | ArrayBufferLike
// Here, no Input0D<Element>[], as direct entries and array buffers should not be mixed up in an entry
export type Input1D<Element> = readonly Element[] | readonly ArrayBufferLike[] | ArrayBufferLike
export type Input2D<Element> = readonly Input1D<Element>[] | ArrayBufferLike
export type Input3D<Element> = readonly Input2D<Element>[] | ArrayBufferLike
export type Input4D<Element> = readonly Input3D<Element>[] | ArrayBufferLike
export type InputXD<Element = any, SizesSpec extends readonly any[] = any[]> = SizesSpec extends []
	? Input0D<Element>
	: SizesSpec extends [any]
		? Input1D<Element>
		: SizesSpec extends [any, any]
			? Input2D<Element>
			: SizesSpec extends [any, any, any]
				? Input3D<Element>
				: SizesSpec extends [any, any, any, any]
					? Input4D<Element>
					: unknown
export type AnyInput = Input0D<any> | Input1D<any> | Input2D<any> | Input3D<any>

export type ValuedBuffable<
	Inferences extends AnyInference = AnyInference,
	Element = any,
	SizesSpec extends readonly SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	ElementSizeSpec extends readonly SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
> = {
	buffable: Buffable<Inferences, Element, SizesSpec, ElementSizeSpec>
	value: InputXD<Element, SizesSpec>
}

/**
 * Interface is needed for type inference
 */
export interface Buffable<
	Inferences extends AnyInference = any,
	Element = any,
	SizesSpec extends readonly SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	ElementSizeSpec extends readonly SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
> {
	readonly size: SizesSpec
	readonly elementSize: ElementSizeSpec
	toArrayBuffer(
		data: InputXD<Element, SizesSpec>,
		inferences: Inferences,
		reason: string,
		reasons: Record<string, string>
	): ArrayBuffer
	value(
		v: InputXD<Element, SizesSpec>
	): ValuedBuffable<Inferences, Element, SizesSpec, ElementSizeSpec>
	readonly wgslSpecification: string
	readArrayBuffer(
		buffer: ArrayBuffer,
		inferences: AnyInference
	): BufferReader<Element, NumericSizesSpec<SizesSpec>>
	elementByteSize(inferences: Inferences): number
}
