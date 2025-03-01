import { Float16Array } from '@petamoriken/float16'
import { InferenceValidationError } from '../types'

export type NumericSizesSpec<SizesSpec extends readonly any[]> = {
	[K in keyof SizesSpec]: number
}

export type TypedArrayConstructor<ArrayBufferLike> = {
	new (content: number[] | number): ArrayBufferLike
	new (ab: ArrayBuffer): ArrayBufferLike
	BYTES_PER_ELEMENT: number
}
export type TypedArray = Float32Array | Float16Array | Uint32Array | Int32Array

export type TypedArrayXD<T extends TypedArray = TypedArray> = T & {
	elementSize: number
	size: number[]
}
export type TypedArray0D<T extends TypedArray = TypedArray> = T & { elementSize: number; size: [] }
export type TypedArray1D<T extends TypedArray = TypedArray> = T & {
	elementSize: number
	size: [number]
}
export type TypedArray2D<T extends TypedArray = TypedArray> = T & {
	elementSize: number
	size: [number, number]
}
export type TypedArray3D<T extends TypedArray = TypedArray> = T & {
	elementSize: number
	size: [number, number, number]
}
export type TypedArray4D<T extends TypedArray = TypedArray> = T & {
	elementSize: number
	size: [number, number, number, number]
}

export function dimensionedArray(typedArray: TypedArray): TypedArray0D
export function dimensionedArray(typedArray: TypedArray, size: [number]): TypedArray1D
export function dimensionedArray(typedArray: TypedArray, size: [number, number]): TypedArray2D
export function dimensionedArray(
	typedArray: TypedArray,
	size: [number, number, number]
): TypedArray3D
export function dimensionedArray(
	typedArray: TypedArray,
	size: [number, number, number, number]
): TypedArray4D
export function dimensionedArray(
	typedArray: TypedArray,
	size:
		| []
		| [number]
		| [number, number]
		| [number, number, number]
		| [number, number, number, number] = [],
	expectedElementSize?: number
): TypedArrayXD {
	const elementSize = typedArray.length / (size as number[]).reduce((a, b) => a * b, 1)
	if (
		![1, 2, 3, 4, 6, 8, 9, 12, 16].includes(elementSize) ||
		(expectedElementSize && elementSize !== expectedElementSize)
	)
		throw new InferenceValidationError(
			`Array size mismatch: With a length of ${typedArray.length} a size-${size.join('x')} array/texture would have an element size of ${elementSize}`
		)
	return Object.assign(typedArray, {
		elementSize,
		size,
	})
}

export function isTypedArray(value: any): value is TypedArray {
	return (
		value instanceof Float32Array ||
		value instanceof Float16Array ||
		value instanceof Uint32Array ||
		value instanceof Int32Array
	)
}
export function isTypedArrayXD(
	value: any
): value is TypedArray0D | TypedArray1D | TypedArray2D | TypedArray3D | TypedArray4D {
	return (
		isTypedArray(value) && 'size' in value && Array.isArray(value.size) && value.size.length <= 4
	)
}
export function isTypedArray2D(value: any): value is TypedArray2D {
	return (
		isTypedArray(value) && 'size' in value && Array.isArray(value.size) && value.size.length === 2
	)
}
export function isTypedArray3D(value: any): value is TypedArray3D {
	return (
		isTypedArray(value) && 'size' in value && Array.isArray(value.size) && value.size.length === 3
	)
}
export function isTypedArray4D(value: any): value is TypedArray3D {
	return (
		isTypedArray(value) && 'size' in value && Array.isArray(value.size) && value.size.length === 4
	)
}
