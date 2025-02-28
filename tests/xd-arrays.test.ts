import { expect } from 'chai'
import createWebGpGpu, {
	inference,
	InferenceValidationError,
	ParameterError,
	Vector2,
	type RootWebGpGpu,
	f32,
	u32,
	vec2f,
	inputs,
	commons,
	outputs,
} from 'webgpgpu'

let webGpGpu: RootWebGpGpu

before(async () => {
	webGpGpu = await createWebGpGpu()
})
after(() => {
	webGpGpu.dispose()
})
describe('arrays', () => {
	it('output', async () => {
		const kernel = webGpGpu
			.input({ a: f32.array('threads.x'), b: f32.array('threads.y') })
			.output({ m: f32.array('threads.y', 'threads.x') })
			.kernel(/*wgsl*/ `
		let stride = vec2u(1, threads.x);
		m[dot(thread.xy, stride)] = a[thread.x]*b[thread.y];
		`)
		const { m } = await kernel({ a: [1, 2], b: [3, 5, 7] })
		expect(m.flat()).to.deep.equal([3, 6, 5, 10, 7, 14])
		expect(m.at(1, 0)).to.equal(5)
		expect(m.at(0, 1)).to.equal(6)
	})
})
