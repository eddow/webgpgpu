import type { Float16Array } from '@petamoriken/float16'

export type InputXD<Element, SizesSpec extends readonly any[]> =
	| (SizesSpec extends [any, ...infer Rest] ? ArrayLike<InputXD<Element, Rest>> : Element)
	| ArrayBuffer
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
