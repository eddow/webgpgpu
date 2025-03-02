import { expect } from 'chai'
import createWebGpGpu, {
	type RootWebGpGpu,
	f32,
	vec2f,
	mat2x3f,
	vec3u,
	mat3x2f,
	mat2x2f,
} from 'webgpgpu'
import { multiplication } from './xd.data'

let webGpGpu: RootWebGpGpu

before(async () => {
	webGpGpu = await createWebGpGpu()
})
after(() => {
	webGpGpu.dispose()
})
describe('xd-arrays', () => {
	for (const { a, b, r } of multiplication.f32)
		it('output - f32', async () => {
			const kernel = webGpGpu
				.input({ a: f32.array('threads.x'), b: f32.array('threads.y') })
				.output({ m: f32.array('threads.y', 'threads.x') })
				.kernel(/*wgsl*/ `
		let stride = vec2u(threads.y, 1);
		m[dot(thread.xy, stride)] = a[thread.x]*b[thread.y];
		`)
			const { m } = await kernel({ a, b })
			expect(m.flat()).to.deepArrayEqual(r)
		})
	for (const { a, b, r } of multiplication.vec2f)
		it('output - vec2f', async () => {
			const kernel = webGpGpu
				.input({ a: vec2f.array('threads.x'), b: vec2f.array('threads.y') })
				.output({ m: vec2f.array('threads.y', 'threads.x') })
				.kernel(/*wgsl*/ `
let stride = vec2u(threads.y, 1);
m[dot(thread.xy, stride)] = a[thread.x]*b[thread.y];
		`)
			const { m } = await kernel({ a, b })
			expect(m.flat()).to.deepArrayEqual(r)
		})
	for (const { a, b, r } of multiplication.vec3u)
		it('output - vec3u', async () => {
			const kernel = webGpGpu
				.input({ a: vec3u.array('threads.x'), b: vec3u.array('threads.y') })
				.output({ m: vec3u.array('threads.y', 'threads.x') })
				.kernel(/*wgsl*/ `
		let stride = vec2u(threads.y, 1);
		m[dot(thread.xy, stride)] = a[thread.x]*b[thread.y];
		`)
			const { m } = await kernel({ a, b })
			expect(m.flat()).to.deepArrayEqual(r)
		})
	for (const { a, b, r } of multiplication.mat2x)
		it('output - mat2x', async () => {
			const kernel = webGpGpu
				.input({ a: mat3x2f.array('threads.x'), b: mat2x3f.array('threads.y') })
				.output({ m: mat2x2f.array('threads.y', 'threads.x') })
				.kernel(/*wgsl*/ `
		let stride = vec2u(threads.y, 1);
		m[dot(thread.xy, stride)] = a[thread.x]*b[thread.y];
		`)
			const { m } = await kernel({ a, b })
			expect(m.flat()).to.deepArrayEqual(r)
		})
})
