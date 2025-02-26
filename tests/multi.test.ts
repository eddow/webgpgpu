import { expect } from 'chai'
import createWebGpGpu, {
	inference,
	InferenceValidationError,
	ParameterError,
	Vector2,
	type WebGpGpu,
	f32,
	u32,
	vec2f,
	inputs,
} from 'webgpgpu'

let webGpGpu: WebGpGpu

before(async () => {
	webGpGpu = await createWebGpGpu()
})
after(() => {
	webGpGpu.dispose()
})
describe('half-size convert', () => {
	// .array(...).transform(...)
	it('todo', () => {
		expect(true).to.be.true
	})
})
