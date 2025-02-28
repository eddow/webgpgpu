import { type AnyInference, type SizeSpec, assertSize } from '../inference'
import type { NumericSizesSpec } from './arrays'
import type { BufferReader } from './io'

export type Input0D<Element> = Element | ArrayBufferLike
export type Input1D<Element> = Input0D<Element>[] | ArrayBufferLike
export type Input2D<Element> = Input1D<Element>[] | ArrayBufferLike
export type Input3D<Element> = Input2D<Element>[] | ArrayBufferLike
export type Input4D<Element> = Input3D<Element>[] | ArrayBufferLike
export type InputXD<Element = any, SizesSpec extends any[] = any[]> = SizesSpec extends []
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

type ValidateSizeSpec<
	Inferences extends AnyInference,
	SizesSpec,
> = SizesSpec extends SizeSpec<Inferences>[] ? unknown : never
export type ValuedBuffable<
	Inferences extends AnyInference = AnyInference,
	Element = any,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	ElementSizeSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
> = {
	buffable: Buffable<Inferences, Element, SizesSpec, ElementSizeSpec>
	value: InputXD<Element, SizesSpec>
} // & ValidateSizeSpec<Inferences, SizesSpec>

/**
 * Interface is needed for type inference
 */
export interface Buffable<
	Inferences extends AnyInference = any,
	Element = any,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	ElementSizeSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
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
