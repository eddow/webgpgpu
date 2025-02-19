import {
	type GpGpuData,
	GpGpuXFloat16,
	GpGpuXFloat32,
	GpGpuXInt32,
	GpGpuXUint32,
	type TypedArray,
} from './dataTypes'
import { WebGpGpu } from './webgpgpu'

// #region GpGpu1x

export const f32 = new GpGpuXFloat32<number>(1, 'f32')
export const u32 = new GpGpuXUint32<number>(1, 'u32')
export const i32 = new GpGpuXInt32<number>(1, 'i32')

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

export let f16: GpGpuData<TypedArray, number[], number> = f32
export let vec2h: GpGpuData<TypedArray, number[], [number, number]> = vec2f
export let vec3h: GpGpuData<TypedArray, number[], [number, number, number]> = vec3f
export let vec4h: GpGpuData<TypedArray, number[], [number, number, number, number]> = vec4f
export let mat2x2h: GpGpuData<TypedArray, number[], [number, number, number, number]> = mat2x2f
export let mat2x3h: GpGpuData<
	TypedArray,
	number[],
	[number, number, number, number, number, number]
> = mat2x3f
export let mat2x4h: GpGpuData<
	TypedArray,
	number[],
	[number, number, number, number, number, number, number, number]
> = mat2x4f
export let mat3x2h: GpGpuData<
	TypedArray,
	number[],
	[number, number, number, number, number, number]
> = mat3x2f
export let mat3x3h: GpGpuData<
	TypedArray,
	number[],
	[number, number, number, number, number, number, number, number, number]
> = mat3x3f
export let mat3x4h: GpGpuData<
	TypedArray,
	number[],
	[number, number, number, number, number, number, number, number, number, number, number, number]
> = mat3x4f
export let mat4x2h: GpGpuData<
	TypedArray,
	number[],
	[number, number, number, number, number, number, number, number]
> = mat4x2f
export let mat4x3h: GpGpuData<
	TypedArray,
	number[],
	[number, number, number, number, number, number, number, number, number, number, number, number]
> = mat4x3f
export let mat4x4h: GpGpuData<
	TypedArray,
	number[],
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
WebGpGpu.root.then((gpu) => {
	if (gpu!.device.features.has('f16')) {
		f16 = new GpGpuXFloat16(1, 'f16')
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
			]
		>(12, 'mat3x4<f16>')
		mat4x2h = new GpGpuXFloat16<[number, number, number, number, number, number, number, number]>(
			8,
			'mat4x2<f16>'
		)
		mat4x3h = new GpGpuXFloat16<
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
			]
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
	}
})
