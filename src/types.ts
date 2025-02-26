import type { Float16Array } from '@petamoriken/float16'

export class WebGpGpuError extends Error {}

export class InferenceValidationError extends WebGpGpuError {
	name = 'InferenceValidationError'
}

export class ParameterError extends WebGpGpuError {
	name = 'ParameterError'
}

export class CompilationError extends WebGpGpuError {
	name = 'CompilationError'
	constructor(public cause: readonly GPUCompilationMessage[]) {
		super('Compilation error')
	}
}

export type TypedArrayConstructor<TArray> = {
	new (content: number[] | number): TArray
	new (ab: ArrayBuffer): TArray
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

export type Input0D<Element, TArray extends TypedArray = TypedArray> = Element | TArray
export type Input1D<Element, TArray extends TypedArray = TypedArray> =
	| Input0D<Element, TArray>[]
	| TArray
export type Input2D<Element, TArray extends TypedArray = TypedArray> =
	| Input1D<Element, TArray>[]
	| TArray
export type Input3D<Element, TArray extends TypedArray = TypedArray> =
	| Input2D<Element, TArray>[]
	| TArray
export type Input4D<Element, TArray extends TypedArray = TypedArray> =
	| Input3D<Element, TArray>[]
	| TArray
export type InputXD<
	Element = any,
	SizesSpec extends number[] = number[],
	TArray extends TypedArray = TypedArray,
> = SizesSpec extends []
	? Input0D<Element, TArray>
	: SizesSpec extends [number]
		? Input1D<Element, TArray>
		: SizesSpec extends [number, number]
			? Input2D<Element, TArray>
			: SizesSpec extends [number, number, number]
				? Input3D<Element, TArray>
				: SizesSpec extends [number, number, number, number]
					? Input4D<Element, TArray>
					: unknown
export type AnyInput = Input0D<any> | Input1D<any> | Input2D<any> | Input3D<any>

export type WorkSize = [number, number, number]

export function mapEntries<From, To, T extends { [key: string]: From }>(
	obj: T,
	fn: (value: From, key: PropertyKey) => To
): { [key: string]: To } {
	return Object.fromEntries(
		Object.entries(obj).map(([key, value]: [PropertyKey, unknown]) => [key, fn(value as From, key)])
	)
}
