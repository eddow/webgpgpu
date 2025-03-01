import type { Input1D, mat2x3, mat3x2, vec2, vec3 } from 'webgpgpu'

type NumbersInArrays =
	| number
	| readonly number[]
	| readonly number[][]
	| readonly number[][][]
	| readonly number[][][][]
	| readonly number[][][][][]
	| readonly number[][][][][][]
type Mul<A, B> = { a: Input1D<A>; b: Input1D<B>; r: NumbersInArrays }
export const multiplication: {
	f32: Mul<number, number>[]
	vec2f: Mul<vec2, vec2>[]
	vec3u: Mul<vec3, vec3>[]
	mat2x: Mul<mat3x2, mat2x3>[]
} = {
	f32: [
		{
			a: [1, 2],
			b: [3, 5, 7],
			r: [3, 5, 7, 6, 10, 14],
		},
	],
	vec2f: [
		{
			a: [
				[1, 2],
				[3, 7],
			],
			b: [
				[3, 5],
				[7, 11],
				[13, 17],
			],
			r: [
				[3, 10],
				[7, 22],
				[13, 34],
				[9, 35],
				[21, 77],
				[39, 119],
			],
		},
	],
	vec3u: [
		{
			a: [
				[1, 2, 3],
				[4, 5, 6],
			],
			b: [
				[7, 8, 9],
				[10, 11, 12],
				[13, 14, 15],
			],
			r: [
				[7, 16, 27],
				[10, 22, 36],
				[13, 28, 45],
				[28, 40, 54],
				[40, 55, 72],
				[52, 70, 90],
			],
		},
	],
	mat2x: [
		{
			a: [
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
			],
			b: [
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
			],
			r: [
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
		},
	],
}
