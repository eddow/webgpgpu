import { Float16Array } from '@petamoriken/float16'
import type { TypedArray } from './arrays'
import { type AtomicAccessor, type GpGpuSingleton, MappedAtomic } from './mapped'

/* padding:
Type	Intended Size	Actual Padded Size	Extra Padding?
vec2<f32>	8B	✅ 8B	No
vec3<f32>	12B	⚠️ 16B	Yes (4B)
vec4<f32>	16B	✅ 16B	No
mat2x3f	24B	⚠️ 32B	Yes (2×4B)
mat3x3f	36B	⚠️ 48B	Yes (3×4B)
mat4x3f	48B	⚠️ 64B	Yes (4×4B)
*/
type lt2<T> = [T, T]
type lt3<T> = [T, T, T]
type lt4<T> = [T, T, T, T]
export type vec2 = lt2<number>
export type vec3 = lt3<number>
export type vec4 = lt4<number>
export type mat2x2 = lt2<vec2>
export type mat2x3 = lt2<vec3>
export type mat2x4 = lt2<vec4>
export type mat3x2 = lt3<vec2>
export type mat3x3 = lt3<vec3>
export type mat3x4 = lt3<vec4>
export type mat4x2 = lt4<vec2>
export type mat4x3 = lt4<vec3>
export type mat4x4 = lt4<vec4>

const vec = <T extends number[]>(n: number): AtomicAccessor<T> => ({
	// @ts-expect-error TypedArray -> [...]
	read: (typedArray, pos) => typedArray.subarray(pos, pos + n),
	write: (typedArray, pos, value) => typedArray.set(value.slice(0, n), pos),
})
const mat = <T extends number[][]>(cols: number, rows: number): AtomicAccessor<T> => {
	const positions: number[] = []
	const colSize = rows === 3 ? 4 : rows
	for (let c = 0; c < cols; c++) positions.push(colSize * c)
	return {
		// @ts-expect-error TypedArray -> [...]
		read: (typedArray, pos) => positions.map((p) => typedArray.subarray(p + pos, p + pos + rows)),
		write: (typedArray, pos, value) => {
			for (let c = 0; c < cols; c++) typedArray.set(value[c].slice(0, rows), pos + colSize * c)
		},
	}
}
const oneF32Type = <T>(
	elementSize: number,
	wgslSpecification: string,
	elementAccessor: AtomicAccessor<T>
) =>
	new MappedAtomic<Float32Array, T>(
		Float32Array,
		elementSize,
		wgslSpecification.replace('#', 'f'),
		elementAccessor
	)
const typeTriplet = <T>(
	elementSize: number,
	wgslSpecification: string,
	elementAccessor: AtomicAccessor<T>
) => [
	oneF32Type(elementSize, wgslSpecification, elementAccessor),
	new MappedAtomic<Uint32Array, T>(
		Uint32Array,
		elementSize,
		wgslSpecification.replace('#', 'u'),
		elementAccessor
	),
	new MappedAtomic<Int32Array, T>(
		Int32Array,
		elementSize,
		wgslSpecification.replace('#', 'i'),
		elementAccessor
	),
]

export const [f32, u32, i32] = typeTriplet<number>(1, '#32', {
	read: (typedArray, index) => typedArray.at(index)!,
	write: (typedArray, index, value) => typedArray.set([value], index),
	writeMany: (typedArray, index, values) => typedArray.set(values, index),
})

export const [vec2f, vec2u, vec2i] = typeTriplet<vec2>(2, 'vec2#', vec(2))
export const [vec3f, vec3u, vec3i] = typeTriplet<vec3>(4, 'vec3#', vec(3))
export const [vec4f, vec4u, vec4i] = typeTriplet<vec4>(4, 'vec4#', vec(4))
export const mat2x2f = oneF32Type<mat2x2>(4, 'mat2x2#', mat(2, 2))
export const mat2x3f = oneF32Type<mat2x3>(8, 'mat2x3#', mat(2, 3))
export const mat2x4f = oneF32Type<mat2x4>(8, 'mat2x4#', mat(2, 4))
export const mat3x2f = oneF32Type<mat3x2>(6, 'mat3x2#', mat(3, 2))
export const mat3x3f = oneF32Type<mat3x3>(12, 'mat3x3#', mat(3, 3))
export const mat3x4f = oneF32Type<mat3x4>(12, 'mat3x4#', mat(3, 4))
export const mat4x2f = oneF32Type<mat4x2>(8, 'mat4x2#', mat(4, 2))
export const mat4x3f = oneF32Type<mat4x3>(16, 'mat4x3#', mat(4, 3))
export const mat4x4f = oneF32Type<mat4x4>(16, 'mat4x4#', mat(4, 4))

// #region f16
// Only exist as vec#h shape
export let vec2h: MappedAtomic<TypedArray, vec2> = vec2f
export let vec3h: MappedAtomic<TypedArray, vec3> = vec3f
export let vec4h: MappedAtomic<TypedArray, vec4> = vec4f

function structAccessor<T extends Record<string, number>>(alphabet: string): AtomicAccessor<T> {
	const letters = alphabet.split('')
	return {
		read(typedArray, index) {
			return Object.fromEntries(letters.map((l, i) => [l, typedArray.at(index + i)!])) as T
		},
		write(typedArray, index, value) {
			for (let i = 0; i < letters.length; i++) typedArray.set([value[letters[i]]], index + i)
		},
	}
}

const xyAcc = structAccessor<{ x: number; y: number }>('xy')
const xyzAcc = structAccessor<{ x: number; y: number; z: number }>('xyz')
const xyzwAcc = structAccessor<{ x: number; y: number; z: number; w: number }>('xyzw')
const rgbAcc = structAccessor<{ r: number; g: number; b: number }>('rgb')
const rgbaAcc = structAccessor<{ r: number; g: number; b: number; a: number }>('rgba')

export let Vector2: MappedAtomic<TypedArray, { x: number; y: number }> = vec2f.transform(xyAcc)
export let Vector3: MappedAtomic<TypedArray, { x: number; y: number; z: number }> =
	vec3f.transform(xyzAcc)
export let Vector4: MappedAtomic<TypedArray, { x: number; y: number; z: number; w: number }> =
	vec4f.transform(xyzwAcc)
export let RGB: MappedAtomic<TypedArray, { r: number; g: number; b: number }> =
	vec3f.transform(rgbAcc)
export let RGBA: MappedAtomic<TypedArray, { r: number; g: number; b: number; a: number }> =
	vec4f.transform(rgbaAcc)
let f16Activated = false
export function activateF16(available: boolean) {
	if (f16Activated) return
	f16Activated = true
	if (!available) return
	vec2h = new MappedAtomic(Float16Array, 2, 'vec2h', vec(2))
	vec3h = new MappedAtomic(Float16Array, 4, 'vec3h', vec(3))
	vec4h = new MappedAtomic(Float16Array, 4, 'vec4h', vec(4))
	Vector2 = vec2h.transform(xyAcc)
	Vector3 = vec3h.transform(xyzAcc)
	Vector4 = vec4h.transform(xyzwAcc)
	RGB = vec3h.transform(rgbAcc)
	RGBA = vec4h.transform(rgbaAcc)
}

// #endregion f16
