import {
	type GpGpuSingleton,
	GpGpuXFloat16,
	GpGpuXFloat32,
	GpGpuXInt32,
	GpGpuXUint32,
} from './buffable'

// #region GpGpu0x
/* TODO: padding + actual shape (eg [[number, number], [number, number]] instead of [number, number, number, number])
Type	Intended Size	Actual Padded Size	Extra Padding?
vec2<f32>	8B	✅ 8B	No
vec3<f32>	12B	⚠️ 16B	Yes (4B)
vec4<f32>	16B	✅ 16B	No
mat2x3f	24B	⚠️ 32B	Yes (2×4B)
mat3x3f	36B	⚠️ 48B	Yes (3×4B)
mat4x3f	48B	⚠️ 64B	Yes (4×4B)
*/
type vec2 = [number, number]
type vec3 = [number, number, number]
type vec4 = [number, number, number, number]
type vec6 = [number, number, number, number, number, number]
type vec8 = [number, number, number, number, number, number, number, number]
type vec9 = [number, number, number, number, number, number, number, number, number]
type vec12 = [
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
type vec16 = [
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

// #endregion GpGpu0x
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

export const vec2f = new GpGpuXFloat32<vec2>(2, 'vec2f')
export const vec2u = new GpGpuXUint32<vec2>(2, 'vec2u')
export const vec2i = new GpGpuXInt32<vec2>(2, 'vec2i')

export const vec3f = new GpGpuXFloat32<vec3>(3, 'vec3f')
export const vec3u = new GpGpuXUint32<vec3>(3, 'vec3u')
export const vec3i = new GpGpuXInt32<vec3>(3, 'vec3i')

export const vec4f = new GpGpuXFloat32<vec4>(4, 'vec4f')
export const vec4u = new GpGpuXUint32<vec4>(4, 'vec4u')
export const vec4i = new GpGpuXInt32<vec4>(4, 'vec4i')

// #endregion GpGpu1x
// #region GpGpu2x

export const mat2x2f = new GpGpuXFloat32<vec4>(4, 'mat2x2f')
export const mat2x2u = new GpGpuXUint32<vec4>(4, 'mat2x2u')
export const mat2x2i = new GpGpuXInt32<vec4>(4, 'mat2x2i')

export const mat2x3f = new GpGpuXFloat32<vec6>(6, 'mat2x3f')
export const mat2x3u = new GpGpuXUint32<vec6>(6, 'mat2x3u')
export const mat2x3i = new GpGpuXInt32<vec6>(6, 'mat2x3i')

export const mat2x4f = new GpGpuXFloat32<vec8>(8, 'mat2x4f')
export const mat2x4u = new GpGpuXUint32<vec8>(8, 'mat2x4u')
export const mat2x4i = new GpGpuXInt32<vec8>(8, 'mat2x4i')

// #endregion GpGpu2x
// #region GpGpu3x

export const mat3x2f = new GpGpuXFloat32<vec6>(6, 'mat3x2f')
export const mat3x2u = new GpGpuXUint32<vec6>(6, 'mat3x2u')
export const mat3x2i = new GpGpuXInt32<vec6>(6, 'mat3x2i')

export const mat3x3f = new GpGpuXFloat32<vec9>(9, 'mat3x3f')
export const mat3x3u = new GpGpuXUint32<vec9>(9, 'mat3x3u')
export const mat3x3i = new GpGpuXInt32<vec9>(9, 'mat3x3i')

export const mat3x4f = new GpGpuXFloat32<vec12>(12, 'mat3x4f')
export const mat3x4u = new GpGpuXUint32<vec12>(12, 'mat3x4u')
export const mat3x4i = new GpGpuXInt32<vec12>(12, 'mat3x4i')

// #endregion GpGpu3x
// #region GpGpu4x

export const mat4x2f = new GpGpuXFloat32<vec8>(8, 'mat4x2f')
export const mat4x2u = new GpGpuXUint32<vec8>(8, 'mat4x2u')
export const mat4x2i = new GpGpuXInt32<vec8>(8, 'mat4x2i')

export const mat4x3f = new GpGpuXFloat32<vec12>(12, 'mat4x3f')
export const mat4x3u = new GpGpuXUint32<vec12>(12, 'mat4x3u')
export const mat4x3i = new GpGpuXInt32<vec12>(12, 'mat4x3i')

export const mat4x4f = new GpGpuXFloat32<vec16>(16, 'mat4x4f')
export const mat4x4u = new GpGpuXUint32<vec16>(16, 'mat4x4u')
export const mat4x4i = new GpGpuXInt32<vec16>(16, 'mat4x4i')

// #endregion GpGpu4x4
// #region f16
// Only exist as vec#h shape

export let vec2h: GpGpuSingleton<vec2> = vec2f
export let vec3h: GpGpuSingleton<vec3> = vec3f
export let vec4h: GpGpuSingleton<vec4> = vec4f
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
	vec2h = new GpGpuXFloat16<vec2>(2, 'vec2h')
	vec3h = new GpGpuXFloat16<vec3>(3, 'vec3h')
	vec4h = new GpGpuXFloat16<vec4>(4, 'vec4h')
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

// #endregion f16
