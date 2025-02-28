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

export type WorkSize = [number, number, number]

export function mapEntries<From, To, T extends { [key: string]: From }>(
	obj: T,
	fn: (value: From, key: PropertyKey) => To
): { [key: string]: To } {
	return Object.fromEntries(
		Object.entries(obj).map(([key, value]: [PropertyKey, unknown]) => [key, fn(value as From, key)])
	)
}
