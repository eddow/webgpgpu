import {
	Buffable,
	type Input1D,
	type ValuedBuffable,
	f32,
	mat2x2f,
	type mat2x3,
	mat2x3f,
	type mat3x2,
	mat3x2f,
	mat3x3f,
	u32,
	type vec2,
	vec2f,
	type vec3,
	vec3f,
	vec3u,
	type vec4,
	vec4f,
} from 'webgpgpu'
import type { mat3x3 } from '../lib/client'

type Mul<A, B> = { a: Input1D<A>; b: Input1D<B>; r: any[] }

function genNumbers(size: number[], position: { p: number }) {
	if (!size.length) return position.p++
	const rv: any[] = []
	for (let i = 0; i < size[0]; ++i) rv.push(genNumbers(size.slice(1), position))
	return rv
}
function gn<T>(...size: number[]) {
	return genNumbers(size, { p: 1 }) as T[]
}
// input1D*input1D = output2D
export const multiplication2: Record<
	string,
	{ a: ValuedBuffable; b: ValuedBuffable; r: ValuedBuffable }
> = {
	f32: {
		a: f32.array('threads.x').value([1, 2]),
		b: f32.array('threads.y').value([3, 5, 7]),
		r: f32.array('threads.x', 'threads.y').value([
			[3, 5, 7],
			[6, 10, 14],
		]),
	},
	vec2f: {
		a: vec2f.array('threads.x').value([
			[1, 2],
			[3, 7],
		]),
		b: vec2f.array('threads.y').value([
			[3, 5],
			[7, 11],
			[13, 17],
		]),
		r: vec2f.array('threads.x', 'threads.y').value([
			[
				[3, 10],
				[7, 22],
				[13, 34],
			],
			[
				[9, 35],
				[21, 77],
				[39, 119],
			],
		]),
	},
	vec3u: {
		a: vec3u.array('threads.x').value([
			[1, 2, 3],
			[4, 5, 6],
		]),
		b: vec3u.array('threads.y').value([
			[7, 8, 9],
			[10, 11, 12],
			[13, 14, 15],
		]),
		r: vec3u.array('threads.x', 'threads.y').value([
			[
				[7, 16, 27],
				[10, 22, 36],
				[13, 28, 45],
			],
			[
				[28, 40, 54],
				[40, 55, 72],
				[52, 70, 90],
			],
		]),
	},
	mat2x: {
		a: mat3x2f.array('threads.x').value([
			[
				[25, 26],
				[27, 28],
				[29, 30],
			],
			[
				[31, 32],
				[33, 34],
				[35, 36],
			],
		]),
		b: mat2x3f.array('threads.y').value([
			[
				[1, 2, 3],
				[4, 5, 6],
			],
			[
				[7, 8, 9],
				[10, 11, 12],
			],
			[
				[13, 14, 15],
				[16, 17, 18],
			],
		]),
		r: mat2x2f.array('threads.x', 'threads.y').value([
			[
				[
					[166, 172],
					[409, 424],
				],
				[
					[652, 676],
					[895, 928],
				],
				[
					[1138, 1180],
					[1381, 1432],
				],
			],
			[
				[
					[202, 208],
					[499, 514],
				],
				[
					[796, 820],
					[1093, 1126],
				],
				[
					[1390, 1432],
					[1687, 1738],
				],
			],
		]),
	},
}

export const XbYTests = {
	'f32<1>': f32.array('x').value(gn<number>(5)),
	'vec2f<1>': vec2f.array('x').value(gn<vec2>(5, 2)),
	'vec3f<1>': vec3f.array('x').value(gn<vec3>(5, 3)),
	'vec4f<1>': vec4f.array('x').value(gn<vec4>(5, 4)),
	'mat2x3f<1>': mat2x3f.array('x').value(gn<mat2x3>(5, 2, 3)),
	'mat3x2f<1>': mat3x2f.array('x').value(gn<mat3x2>(5, 3, 2)),
	'f32<2>': f32.array('x', 'y').value(gn<number[]>(5, 7)),
	'mat2x3f<2>': mat2x3f.array('x', 'y').value(gn<mat2x3[]>(5, 6, 2, 3)),
	'u32<3>': u32.array('x', 'y', 'z').value(gn<number[][]>(5, 6, 7)),
	'vec3i<4>': vec3f.array('x', 'y', 'z', 'w').value(gn<vec3[][][]>(5, 6, 7, 4, 3)),
	'mat3x3f<4>': mat3x3f.array('x', 'y', 'z', 'w').value(gn<mat3x3[][][]>(5, 6, 7, 4, 3, 3)),
}
