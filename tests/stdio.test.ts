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
describe('overall', () => {
	it('toString', async () => {
		const kernel = webGpGpu.workGroup(42).kernel('')
		expect(kernel).to.exist
		expect(kernel).to.be.a('function')
		//'@group(0) @binding(0) var<uniform> threads : vec3u;'
		expect(kernel.toString()).to.match(
			/@group\(\d+\) @binding\(\d+\) var<uniform> threads : vec3u;/
		)
		expect(kernel.toString()).to.match(
			/@compute @workgroup_size\(42,1,1\)\s*fn main\(@builtin\(global_invocation_id\) thread : vec3u\) {\s*if\(all\(thread < threads\)\) {\s*}\s*}/
			/*new RegExp(`
@compute @workgroup_size(42,1,1)
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
		
	}
}`)*/
		)
	})
	it('no argument', async () => {
		const kernel = webGpGpu.kernel('')
		const empty = await kernel({})
		expect(empty).to.be.an('object').that.is.empty
	})
	it('one input', async () => {
		const kernel = webGpGpu.input({ a: f32 }).kernel('')
		const empty = await kernel({ a: Float32Array.from([43]) })
		expect(empty).to.be.an('object').that.is.empty
	})
	it('one output', async () => {
		const kernel = webGpGpu.output({ output: f32.array('threads.x') }).kernel('output[0] = 42;')
		const { output } = await kernel({}, { 'threads.x': 1 })
		expect(output).to.exist
		expect(output.toArray()).to.deep.equal([42])
	})
})
describe('inputs', () => {
	it('an uniform - TypedArray', async () => {
		const kernel = webGpGpu
			.bind(inputs({ a: f32 }))
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a;')
		const { output } = await kernel({ a: Float32Array.from([43]) }, { 'threads.x': 1 })
		expect(output).to.exist
		expect(output.toArray()).to.deep.equal([43])
	})
	it('an uniform - value', async () => {
		const kernel = webGpGpu
			.input({ a: f32 })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a;')
		const { output } = await kernel({ a: 44 }, { 'threads.x': 1 })
		expect(output.toArray()).to.deep.equal([44])
	})
	it('a float array - TypedArray', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x]*3;')
		const { output } = await kernel({ a: Float32Array.from([11, 12, 13]) }, { 'threads.x': 3 })
		expect(output.toArray()).to.deep.equal([33, 36, 39])
	})
	it('a float array - value', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x]*3;')
		const { output } = await kernel({ a: [11, 12, 13] }, { 'threads.x': 3 })
		expect(output.toArray()).to.deep.equal([33, 36, 39])
	})
	it('a vec2 array - TypedArray', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: vec2f.array(3) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const { output } = await kernel(
			{ a: Float32Array.from([1, 11, 2, 12, 3, 13]) },
			{ 'threads.x': 3 }
		)
		expect(output.toArray()).to.deep.equal([36, 42, 48])
	})
	it('a vec2 array - value', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: vec2f.array(3) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const { output } = await kernel(
			{
				a: [
					[1, 11],
					[2, 12],
					[3, 13],
				],
			},
			{ 'threads.x': 3 }
		)
		expect(output.toArray()).to.deep.equal([36, 42, 48])
	})
	it('a vec2 array - transform', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: Vector2.array(3) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const { output } = await kernel(
			{
				a: [
					{ x: 1, y: 11 },
					{ x: 2, y: 12 },
					{ x: 3, y: 13 },
				],
			},
			{ 'threads.x': 3 }
		)
		expect(output.toArray()).to.deep.equal([36, 42, 48])
	})
	it('two arguments', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3), b: f32 })
			.bind(outputs({ output: f32.array('threads.x') }))
			.kernel('output[thread.x] = a[thread.x] * b;')
		const { output } = await kernel(
			{
				a: Float32Array.from([1, 2, 3]),
				b: 5,
			},
			{ 'threads.x': 3 }
		)
		expect(output.toArray()).to.deep.equal([5, 10, 15])
	})
})
describe('infers size', () => {
	it('infer', async () => {
		const kernel = webGpGpu
			.bind(
				commons({
					a: f32.array('threads.x').value([1, 2, 3]),
					b: f32.array('threads.y').value([4, 5, 6, 7, 8]),
				})
			)
			.kernel('')
		expect(kernel.toString()).to.match(/@workgroup_size\(4,8,/)
	})
	it('custom', async () => {
		const kernel = webGpGpu
			.bind(inference({ custom: [undefined, undefined] }))
			.bind(
				commons({
					a: u32.array('threads.x').value([1, 2, 3]),
					b: f32.array('custom.y').value([4, 5, 6, 7, 8]),
				})
			)
			.bind(outputs({ output: u32.array('threads.x') }))
			.kernel('output[thread.x] = a[thread.x] + custom.y;')
		const { output } = await kernel({})
		expect(output.toArray()).to.deep.equal([6, 7, 8])
	})
	it('assert infer fails', async () => {
		expect(() =>
			webGpGpu
				.bind(inference({ custom: [undefined, 8] }))
				.common({
					b: f32.array('custom.y').value([4, 5, 6, 7, 8]),
				})
				.kernel('')
		).to.throw(InferenceValidationError)
	})
	it('assert infer exist', async () => {
		const t = f32.array('qwe').value([4, 5, 6, 7, 8])

		expect(() =>
			webGpGpu.common({
				// @ts-expect-error 'other' is not an inference key
				a: f32.array('other').value([4, 5, 6, 7, 8]),
			})
		).to.throw(ParameterError)
	})
	it('assert', async () => {
		expect(() =>
			webGpGpu.common({
				a: f32.array('threads.x').value([1, 2, 3]),
				b: f32.array('threads.x').value([4, 5, 6, 7, 8]),
			})
		).to.throw(InferenceValidationError)
	})
	it('effect - common', async () => {
		const kernel = webGpGpu
			.common({ a: f32.array('threads.x').value([1, 2, 3]) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x]+3.;')

		const { output } = await kernel({})
		expect(output.toArray()).to.deep.equal([4, 5, 6])
	})
	it('effect - input', async () => {
		const kernel = webGpGpu
			.input({ a: f32.array('threads.x') })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x]+2.;')
		const { output } = await kernel({ a: [1, 2, 3] })
		expect(output.toArray()).to.deep.equal([3, 4, 5])
	})
})
describe('diverse', () => {
	it('defines', async () => {
		const kernel = webGpGpu
			.input({ a: f32.array('threads.x') })
			.common({ b: f32.array('threads.x').value([2, 4, 6]) })
			.define('fn myFunc(a: f32, b: f32) -> f32 { return a + b; }')
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = myFunc(a[thread.x], b[thread.x]);')
		const { output } = await kernel({ a: Float32Array.from([1, 2, 3]) })
		expect(output.toArray()).to.deep.equal([3, 6, 9])
	})
	it('name conflict', async () => {
		const kernel = webGpGpu
			.input({ a: f32.array('threads.x') })
			.common({ b: f32.array('threads.x').value([2, 4, 6]) })
		expect(() => kernel.input({ a: f32.array('threads.y') })).to.throw()
		expect(() => kernel.input({ b: f32.array('threads.y') })).to.throw()
	})
	it('manages big buffers', async () => {
		// TODO: Do something with the error ... Throw ? Catch it first
		const length = 0x400000
		const kernel = webGpGpu.output({ output: u32.array('threads.x') }).kernel(/*wgsl*/ `
let modX = thread.x % 453;
output[thread.x] = modX * modX;
`)
		const { output } = await kernel({}, { 'threads.x': length })
		expect(output).to.exist
		const array = output.toArray()
		expect(array.length).to.equal(length)
	})
})
describe('outputs', () => {
	it('todo', () => {
		expect(true).to.be.true
	})
})
describe('half-size convert', () => {
	// .array(...).transform(...)
	it('todo', () => {
		expect(true).to.be.true
	})
})
