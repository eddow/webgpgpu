import {
	type GpGpuSingleton,
	GpGpuXFloat16,
	GpGpuXFloat32,
	GpGpuXInt32,
	GpGpuXUint32,
} from './dataTypes'

// #region GpGpu1x

export const f32 = new GpGpuXFloat32<number>(
	1,
	'f32',
	(n) => [n],
	(a) => a[0]
)
export const u32 = new GpGpuXUint32<number>(
	1,
	'u32',
	(n) => [n],
	(a) => a[0]
)
export const i32 = new GpGpuXInt32<number>(
	1,
	'i32',
	(n) => [n],
	(a) => a[0]
)

export const vec2f = new GpGpuXFloat32<[number, number]>(2, 'vec2<f32>')
export const vec2u = new GpGpuXUint32<[number, number]>(2, 'vec2<u32>')
export const vec2i = new GpGpuXInt32<[number, number]>(2, 'vec2<i32>')

export const vec3f = new GpGpuXFloat32<[number, number, number]>(3, 'vec3<f32>')
export const vec3u = new GpGpuXUint32<[number, number, number]>(3, 'vec3<u32>')
export const vec3i = new GpGpuXInt32<[number, number, number]>(3, 'vec3<i32>')

export const vec4f = new GpGpuXFloat32<[number, number, number, number]>(4, 'vec4<f32>')
export const vec4u = new GpGpuXUint32<[number, number, number, number]>(4, 'vec4<u32>')
export const vec4i = new GpGpuXInt32<[number, number, number, number]>(4, 'vec4<i32>')

// #endregion GpGpu1x
// #region GpGpu2x

export const mat2x2f = new GpGpuXFloat32<[number, number, number, number]>(4, 'mat2x2<f32>')
export const mat2x2u = new GpGpuXUint32<[number, number, number, number]>(4, 'mat2x2<u32>')
export const mat2x2i = new GpGpuXInt32<[number, number, number, number]>(4, 'mat2x2<i32>')

export const mat2x3f = new GpGpuXFloat32<[number, number, number, number, number, number]>(
	6,
	'mat2x3<f32>'
)
export const mat2x3u = new GpGpuXUint32<[number, number, number, number, number, number]>(
	6,
	'mat2x3<u32>'
)
export const mat2x3i = new GpGpuXInt32<[number, number, number, number, number, number]>(
	6,
	'mat2x3<i32>'
)

export const mat2x4f = new GpGpuXFloat32<
	[number, number, number, number, number, number, number, number]
>(8, 'mat2x4<f32>')
export const mat2x4u = new GpGpuXUint32<
	[number, number, number, number, number, number, number, number]
>(8, 'mat2x4<u32>')
export const mat2x4i = new GpGpuXInt32<
	[number, number, number, number, number, number, number, number]
>(8, 'mat2x4<i32>')

// #endregion GpGpu2x
// #region GpGpu3x

export const mat3x2f = new GpGpuXFloat32<[number, number, number, number, number, number]>(
	6,
	'mat3x2<f32>'
)
export const mat3x2u = new GpGpuXUint32<[number, number, number, number, number, number]>(
	6,
	'mat3x2<u32>'
)
export const mat3x2i = new GpGpuXInt32<[number, number, number, number, number, number]>(
	6,
	'mat3x2<i32>'
)

export const mat3x3f = new GpGpuXFloat32<
	[number, number, number, number, number, number, number, number, number]
>(9, 'mat3x3<f32>')
export const mat3x3u = new GpGpuXUint32<
	[number, number, number, number, number, number, number, number, number]
>(9, 'mat3x3<u32>')
export const mat3x3i = new GpGpuXInt32<
	[number, number, number, number, number, number, number, number, number]
>(9, 'mat3x3<i32>')

export const mat3x4f = new GpGpuXFloat32<
	[number, number, number, number, number, number, number, number, number, number, number, number]
>(12, 'mat3x4<f32>')
export const mat3x4u = new GpGpuXUint32<
	[number, number, number, number, number, number, number, number, number, number, number, number]
>(12, 'mat3x4<u32>')
export const mat3x4i = new GpGpuXInt32<
	[number, number, number, number, number, number, number, number, number, number, number, number]
>(12, 'mat3x4<i32>')

// #endregion GpGpu3x
// #region GpGpu4x

export const mat4x2f = new GpGpuXFloat32<
	[number, number, number, number, number, number, number, number]
>(8, 'mat4x2<f32>')
export const mat4x2u = new GpGpuXUint32<
	[number, number, number, number, number, number, number, number]
>(8, 'mat4x2<u32>')
export const mat4x2i = new GpGpuXInt32<
	[number, number, number, number, number, number, number, number]
>(8, 'mat4x2<i32>')

export const mat4x3f = new GpGpuXFloat32<
	[number, number, number, number, number, number, number, number, number, number, number, number]
>(12, 'mat4x3<f32>')
export const mat4x3u = new GpGpuXUint32<
	[number, number, number, number, number, number, number, number, number, number, number, number]
>(12, 'mat4x3<u32>')
export const mat4x3i = new GpGpuXInt32<
	[number, number, number, number, number, number, number, number, number, number, number, number]
>(12, 'mat4x3<i32>')

export const mat4x4f = new GpGpuXFloat32<
	[
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
	]
>(16, 'mat4x4<f32>')
export const mat4x4u = new GpGpuXUint32<
	[
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
	]
>(16, 'mat4x4<u32>')
export const mat4x4i = new GpGpuXInt32<
	[
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
	]
>(16, 'mat4x4<i32>')

// #endregion GpGpu4x4

export let f16: GpGpuSingleton<number> = f32
export let vec2h: GpGpuSingleton<[number, number]> = vec2f
export let vec3h: GpGpuSingleton<[number, number, number]> = vec3f
export let vec4h: GpGpuSingleton<[number, number, number, number]> = vec4f
export let mat2x2h: GpGpuSingleton<[number, number, number, number]> = mat2x2f
export let mat2x3h: GpGpuSingleton<[number, number, number, number, number, number]> = mat2x3f
export let mat2x4h: GpGpuSingleton<
	[number, number, number, number, number, number, number, number]
> = mat2x4f
export let mat3x2h: GpGpuSingleton<[number, number, number, number, number, number]> = mat3x2f
export let mat3x3h: GpGpuSingleton<
	[number, number, number, number, number, number, number, number, number]
> = mat3x3f
export let mat3x4h: GpGpuSingleton<
	[number, number, number, number, number, number, number, number, number, number, number, number]
> = mat3x4f
export let mat4x2h: GpGpuSingleton<
	[number, number, number, number, number, number, number, number]
> = mat4x2f
export let mat4x3h: GpGpuSingleton<
	[number, number, number, number, number, number, number, number, number, number, number, number]
> = mat4x3f
export let mat4x4h: GpGpuSingleton<
	[
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
		number,
	]
> = mat4x4f
export let Vector2: GpGpuSingleton<{ x: number; y: number }> = vec2f.transform(
	(v) => [v.x, v.y],
	(v) => ({ x: v[0], y: v[1] })
)
export let Vector3: GpGpuSingleton<{ x: number; y: number; z: number }> = vec3f.transform(
	(v) => [v.x, v.y, v.z],
	(v) => ({ x: v[0], y: v[1], z: v[2] })
)
export let Vector4: GpGpuSingleton<{ x: number; y: number; z: number; w: number }> =
	vec4f.transform(
		(v) => [v.x, v.y, v.z, v.w],
		(v) => ({ x: v[0], y: v[1], z: v[2], w: v[3] })
	)
export let RGB: GpGpuSingleton<{ r: number; g: number; b: number }> = vec3f.transform(
	(v) => [v.r, v.g, v.b],
	(v) => ({ r: v[0], g: v[1], b: v[2] })
)
export let RGBA: GpGpuSingleton<{ r: number; g: number; b: number; a: number }> = vec4f.transform(
	(v) => [v.r, v.g, v.b, v.a],
	(v) => ({ r: v[0], g: v[1], b: v[2], a: v[3] })
)
let f16Activated = false
export function activateF16(available: boolean) {
	if (f16Activated) return
	f16Activated = true
	if (!available) return
	f16 = new GpGpuXFloat16<number>(
		1,
		'f16',
		(n) => [n],
		(a) => a[0]
	)
	vec2h = new GpGpuXFloat16<[number, number]>(2, 'vec2<f16>')
	vec3h = new GpGpuXFloat16<[number, number, number]>(3, 'vec3<f16>')
	vec4h = new GpGpuXFloat16<[number, number, number, number]>(4, 'vec4<f16>')
	mat2x2h = new GpGpuXFloat16<[number, number, number, number]>(4, 'mat2x2<f16>')
	mat2x3h = new GpGpuXFloat16<[number, number, number, number, number, number]>(6, 'mat2x3<f16>')
	mat2x4h = new GpGpuXFloat16<[number, number, number, number, number, number, number, number]>(
		8,
		'mat2x4<f16>'
	)
	mat3x2h = new GpGpuXFloat16<[number, number, number, number, number, number]>(6, 'mat3x2<f16>')
	mat3x3h = new GpGpuXFloat16<
		[number, number, number, number, number, number, number, number, number]
	>(9, 'mat3x3<f16>')
	mat3x4h = new GpGpuXFloat16<
		[number, number, number, number, number, number, number, number, number, number, number, number]
	>(12, 'mat3x4<f16>')
	mat4x2h = new GpGpuXFloat16<[number, number, number, number, number, number, number, number]>(
		8,
		'mat4x2<f16>'
	)
	mat4x3h = new GpGpuXFloat16<
		[number, number, number, number, number, number, number, number, number, number, number, number]
	>(12, 'mat4x3<f16>')
	mat4x4h = new GpGpuXFloat16<
		[
			number,
			number,
			number,
			number,
			number,
			number,
			number,
			number,
			number,
			number,
			number,
			number,
			number,
			number,
			number,
			number,
		]
	>(16, 'mat4x4<f16>')
	Vector2 = vec2h.transform(
		(v) => [v.x, v.y],
		(v) => ({ x: v[0], y: v[1] })
	)
	Vector3 = vec3h.transform(
		(v) => [v.x, v.y, v.z],
		(v) => ({ x: v[0], y: v[1], z: v[2] })
	)
	Vector4 = vec4h.transform(
		(v) => [v.x, v.y, v.z, v.w],
		(v) => ({ x: v[0], y: v[1], z: v[2], w: v[3] })
	)
	RGB = vec3h.transform(
		(v) => [v.r, v.g, v.b],
		(v) => ({ r: v[0], g: v[1], b: v[2] })
	)
	RGBA = vec4h.transform(
		(v) => [v.r, v.g, v.b, v.a],
		(v) => ({ r: v[0], g: v[1], b: v[2], a: v[3] })
	)
}
