import { Float16Array } from '@petamoriken/float16'
import type { Buffable, NumericSizesSpec } from './dataTypes'

export class ArraySizeValidationError extends Error {}

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

export function dimensionedArray(typedArray: TypedArray, size: []): TypedArray0D
export function dimensionedArray(typedArray: TypedArray, size: [number]): TypedArray1D
export function dimensionedArray(typedArray: TypedArray, size: [number, number]): TypedArray2D
export function dimensionedArray(
	typedArray: TypedArray,
	size: [number, number, number]
): TypedArray3D
export function dimensionedArray(
	typedArray: TypedArray,
	size: [] | [number] | [number, number] | [number, number, number]
): TypedArrayXD {
	const elementSize = typedArray.length / size.reduce((a, b) => a * b, 1)
	if (![1, 2, 3, 4, 6, 8, 9, 12, 16].includes(elementSize))
		throw new ArraySizeValidationError(
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
export function isTypedArrayXD(value: any): value is TypedArray2D | TypedArray3D {
	return (
		isTypedArray(value) && 'size' in value && Array.isArray(value.size) && value.size.length <= 3
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

// TODO: Structs

export type Input0D<Element, TArray extends TypedArray = TypedArray> = Element | TArray
export type Input1D<Element, TArray extends TypedArray = TypedArray> =
	| Input0D<Element, TArray>[]
	| TArray
export type Input2D<Element, TArray extends TypedArray = TypedArray> =
	| Input1D<Element, TArray>[]
	| TypedArray2D<TArray>
export type Input3D<Element, TArray extends TypedArray = TypedArray> =
	| Input2D<Element, TArray>[]
	| TypedArray3D<TArray>
export type InputXD<
	Element,
	SizesSpec extends number[],
	TArray extends TypedArray,
> = SizesSpec extends []
	? Input0D<Element, TArray>
	: SizesSpec extends [number]
		? Input1D<Element, TArray>
		: SizesSpec extends [number, number]
			? Input2D<Element, TArray>
			: SizesSpec extends [number, number, number]
				? Input3D<Element, TArray>
				: unknown
export type AnyInput = Input0D<any> | Input1D<any> | Input2D<any> | Input3D<any>
export const threads = {
	x: Symbol('threads.x'),
	y: Symbol('threads.y'),
	z: Symbol('threads.z'),
} as const
export const threadAxis = {
	[threads.x]: 'x',
	[threads.y]: 'y',
	[threads.z]: 'z',
} as { [key in (typeof threads)[keyof typeof threads]]: keyof typeof threads }
export const dimensionIndices = {
	[threads.x]: 0,
	[threads.y]: 1,
	[threads.z]: 2,
}

export type SizeSpec = number | keyof typeof threadAxis
export type WorkSizeInfer = {
	x?: number
	y?: number
	z?: number
}

export function assertSize(given: number[], expected: SizeSpec[], workSizeInfer: WorkSizeInfer) {
	if (given.length !== expected.length)
		throw new ArraySizeValidationError(
			`Dimension mismatch in size comparison: ${given.length} !== ${expected.length}`
		)
	for (let i = 0; i < given.length; i++) {
		if (typeof expected[i] === 'number') {
			if (expected[i] === given[i]) continue
			throw new ArraySizeValidationError(
				`Size mismatch in on threads.${'xyz'[i]}: ${given[i]} !== ${expected[i] as number}`
			)
		}
		if (!(expected[i] in threadAxis))
			throw new ArraySizeValidationError(
				`Unexpected size specification, expected number of thread axis: ${String(expected[i])}`
			)
		const dIndex = threadAxis[expected[i] as keyof typeof threadAxis]
		if (workSizeInfer[dIndex] === undefined) {
			workSizeInfer[dIndex] = given[i]
			continue
		}
		if (workSizeInfer[dIndex] !== given[i])
			throw new ArraySizeValidationError(
				`Size mismatch on threads.${'xyz'[i]}: ${given[i]} !== ${workSizeInfer[dIndex]}`
			)
	}
}
export function assertElementSize(given: any, expected: number) {
	if (given !== expected)
		throw new ArraySizeValidationError(
			`Element size mismatch: ${given} received while expecting ${expected}`
		)
}

export class SizeInferError extends Error {}
/**
 * Expect all sizes to be inferred, returns them
 */
export function resolvedSize<SS extends SizeSpec[]>(
	size: SS,
	workSizeInfer: WorkSizeInfer
): NumericSizesSpec<SS> {
	return size.map((s) => {
		const rv = typeof s === 'number' ? s : workSizeInfer[threadAxis[s]]
		if (rv === undefined)
			throw new SizeInferError(`Size ${threadAxis[s as keyof typeof threadAxis]} not inferred`)
		return rv
	}) as NumericSizesSpec<SS>
}

/**
 * Expect all sizes to be inferred, returns their product
 */
export function inferSize(size: SizeSpec[], workSizeInfer: WorkSizeInfer) {
	return resolvedSize(size, workSizeInfer).reduce((a, b) => a * b, 1)
}
