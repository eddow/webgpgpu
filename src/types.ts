import type { Float16Array } from '@petamoriken/float16'

export class WebGpGpuError extends Error {}

export class ArraySizeValidationError extends WebGpGpuError {
	name = 'ArraySizeValidationError'
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

export type RequiredAxis = '' | 'x' | 'y' | 'z' | 'xy' | 'xz' | 'yz' | 'xyz'
export type WorkSize = [number] | [number, number] | [number, number, number]
