import { expect } from 'chai'
import createWebGpGpu, {
	type AnyInference,
	type InputXD,
	type RootWebGpGpu,
	type SizeSpec,
	f32,
	type mat2x3,
	mat2x3f,
	type mat3x2,
	mat3x2f,
	type mat3x3,
	mat3x3f,
	mat4x3f,
	resolvedSize,
	u32,
	type vec2,
	vec2f,
	type vec3,
	vec3f,
	vec3i,
	vec3u,
	type vec4,
	vec4f,
} from 'webgpgpu'
import type { mat4x3 } from '../lib/client'

let webGpGpu: RootWebGpGpu

function cpuMul<SizesSpec extends readonly SizeSpec<AnyInference>[]>(
	value: InputXD<number, SizesSpec>,
	mul: number
): any[] | number {
	if (typeof value === 'number') return value * mul
	if (Array.isArray(value)) return value.map((x) => cpuMul(x, mul))
	throw new Error(`Unsupported type ${typeof value}`)
}

function genNumbers(size: number[], position: { p: number }) {
	if (!size.length) return position.p++
	const rv: any[] = []
	for (let i = 0; i < size[0]; ++i) rv.push(genNumbers(size.slice(1), position))
	return rv
}
function gn<T>(...size: number[]) {
	return genNumbers(size, { p: 1 }) as T[]
}

before(async () => {
	webGpGpu = await createWebGpGpu()
})
after(() => {
	webGpGpu.dispose()
})
describe('xd-arrays', () => {
	// multiply two 1D-arrays to create a matrix of all products
	describe('multiply 2D', () => {
		for (const [m, { a, b, r }] of Object.entries({
			f32: {
				a: f32.array('threads.x').value([1, 2]),
				b: f32.array('threads.y').value([3, 5, 7]),
				r: f32.array('threads.x', 'threads.y').value([
					[3, 5, 7],
					[6, 10, 14],
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
		}))
			it(m, async () => {
				const kernel = webGpGpu
					.common({ a: a as any, b: b as any })
					.output({ m: r.buffable })
					.kernel(/*wgsl*/ `
	let stride = vec2u(threads.y, 1);
	m[dot(thread.xy, stride)] = a[thread.x]*b[thread.y];
		`)
				const { m } = await kernel({})
				expect(m).to.deepArrayEqual(r.value as any)
			})
	})
	describe('multiply-K', () => {
		const mul = 2
		for (const [name, { buffable, value }] of Object.entries({
			// mul is `f32` -> only floats
			'f32<1>': f32.array('threads.x').value(gn<number>(7)),
			'vec3f<1>': vec3f.array('threads.x').value(gn<vec3>(7, 3)),
			'mat4x3f<3>': mat4x3f
				.array('threads.x', 'threads.y', 'threads.z')
				.value(gn<mat4x3[][]>(5, 6, 7, 4, 3)),
		}))
			it(name, async () => {
				const kernel = webGpGpu
					.input({ a: buffable, mul: f32 })
					.output({ m: buffable })
					.kernel(/*wgsl*/ `

	let stride = vec3u(threads.y*threads.z, threads.z, 1);
	let index = dot(thread.xyz, stride);
	m[index] = a[index]*mul;
		`)
				const { m } = await kernel({ a: value, mul })
				expect(m).to.deepArrayEqual(cpuMul(value as any, mul) as any[])
			})
	})
	// Write an input array in an ArrayBuffer and read the same from the buffer
	// No WebGpGpu involved here, just buffer translation
	describe('X->buffer->Y: X~Y', () => {
		for (const [name, { buffable, value }] of Object.entries({
			'f32<1>': f32.array('x').value(gn<number>(5)),
			'vec3f<1>': vec3f.array('x').value(gn<vec3>(5, 3)),
			'vec4f<1>': vec4f.array('x').value(gn<vec4>(5, 4)),
			'mat2x3f<1>': mat2x3f.array('x').value(gn<mat2x3>(5, 2, 3)),
			'mat3x2f<1>': mat3x2f.array('x').value(gn<mat3x2>(5, 3, 2)),
			'f32<2>': f32.array('x', 'y').value(gn<number[]>(5, 7)),
			'mat2x3f<2>': mat2x3f.array('x', 'y').value(gn<mat2x3[]>(5, 6, 2, 3)),
			'u32<3>': u32.array('x', 'y', 'z').value(gn<number[][]>(5, 6, 7)),
			'vec3i<4>': vec3i.array('x', 'y', 'z', 'w').value(gn<vec3[][][]>(5, 6, 7, 4, 3)),
			'mat3x3f<4>': mat3x3f.array('x', 'y', 'z', 'w').value(gn<mat3x3[][][]>(5, 6, 7, 4, 3, 3)),
		}))
			it(name, () => {
				const inferences = { x: undefined, y: undefined, z: undefined, w: undefined }
				const array = value as any[]
				// `toArrayBuffer` has to occur first in order to infer size
				const arrayBuffer = buffable.toArrayBuffer(value as any, inferences)
				expect(arrayBuffer.byteLength).to.equal(
					resolvedSize(buffable.size, inferences).reduce(
						// x * y * ...
						(a, c) => a * c,
						// This part takes padding into account.
						buffable.elementByteSize(inferences)
					)
				)
				const reader = buffable.readArrayBuffer(arrayBuffer, inferences)
				expect(reader).to.deepArrayEqual(array)
			})
	})
})
