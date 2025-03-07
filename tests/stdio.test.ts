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
			/@compute @workgroup_size\(42, 1, 1\)\s*fn main\(@builtin\(global_invocation_id\) thread : vec3u\) {\s*if\(all\(thread < threads\)\) {\s*}\s*}/
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
		const empty = await kernel({ a: Float32Array.from([43]).buffer })
		expect(empty).to.be.an('object').that.is.empty
	})
	it('one output', async () => {
		const kernel = webGpGpu.output({ output: f32 }).kernel('output = 42;')
		const { output } = await kernel({})
		expect(output).to.exist
		expect(output.at()).to.equal(42)
	})
	it('one constant', async () => {
		const kernel = webGpGpu
			.output({ output: f32.array('threads.x') })
			.define({
				declarations: 'override myK: f32 = 9;',
			})
			.kernel('output[thread.x] = myK;')
		const { output } = await kernel({})
		expect(output).to.exist
		expect(output).to.deepArrayEqual([9])
	})
	it('one given constant', async () => {
		const kernel = webGpGpu
			.output({ output: f32 })
			.define({
				declarations: 'override myK: f32 = 9;',
			})
			.kernel('output = myK;', { myK: 52 })
		const { output } = await kernel({})
		expect(output).to.exist
		expect(output.at()).to.equal(52)
	})
})
describe('inputs', () => {
	it('a single - TypedArray', async () => {
		const kernel = webGpGpu
			.bind(inputs({ a: f32 }))
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a;')
		const { output } = await kernel({ a: Float32Array.from([43]) })
		expect(output).to.exist
		expect(output).to.deepArrayEqual([43])
	})
	it('a single - value', async () => {
		const kernel = webGpGpu.input({ a: f32 }).output({ output: f32 }).kernel('output = a;')
		const { output } = await kernel({ a: 44 })
		expect(output.at()).to.equal(44)
	})
	it('a float array - TypedArray', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x]*3;')
		const { output } = await kernel(
			{ a: Float32Array.from([11, 12, 13]).buffer },
			{ 'threads.x': 3 }
		)
		expect(output).to.deepArrayEqual([33, 36, 39])
	})
	it('a float array - value', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x]*3;')
		const { output } = await kernel({ a: [11, 12, 13] }, { 'threads.x': 3 })
		expect(output).to.deepArrayEqual([33, 36, 39])
	})
	it('a vec2 array - TypedArray', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: vec2f.array(3) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const { output } = await kernel(
			{ a: Float32Array.from([1, 11, 2, 12, 3, 13]).buffer },
			{ 'threads.x': 3 }
		)
		expect(output).to.deepArrayEqual([36, 42, 48])
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
		expect(output).to.deepArrayEqual([36, 42, 48])
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
		expect(output).to.deepArrayEqual([36, 42, 48])
	})
	it('two arguments', async () => {
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3) })
			.input({ b: f32 })
			.bind(outputs({ output: f32.array('threads.x') }))
			.kernel('output[thread.x] = a[thread.x] * b;')
		const { output } = await kernel(
			{
				a: Float32Array.from([1, 2, 3]).buffer,
				b: 5,
			},
			{ 'threads.x': 3 }
		)
		expect(output).to.deepArrayEqual([5, 10, 15])
	})
	it('array with ArrayBuffer', async () => {
		const kernel = webGpGpu
			.input({ a: f32.array('threads.x', 2) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x*2] + a[thread.x*2+1];')
		const { output } = await kernel({
			a: [[1, 2], Float32Array.from([3, 4]), Float32Array.from([5, 6]).buffer],
		})
		expect(output).to.deepArrayEqual([3, 7, 11])
	})
	it('complex ArrayBuffer', async () => {
		const buffer = new Float32Array([1, 2, 42, 3, 4, 42, 5, 6]).buffer
		const kernel = webGpGpu
			.input({ a: f32.array('threads.x', 2) })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x*2] + a[thread.x*2+1];')
		const { output } = await kernel({
			a: [new DataView(buffer, 0, 8), new DataView(buffer, 12, 8), new DataView(buffer, 24, 8)],
		})
		expect(output).to.deepArrayEqual([3, 7, 11])
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
		expect(kernel.toString()).to.match(/@workgroup_size\(4, 8,/)
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
		expect(output).to.deepArrayEqual([6, 7, 8])
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
		expect(output).to.deepArrayEqual([4, 5, 6])
	})
	it('effect - input', async () => {
		const kernel = webGpGpu
			.input({ a: f32.array('threads.x') })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x]+2.;')
		const { output } = await kernel({ a: [1, 2, 3] })
		expect(output).to.deepArrayEqual([3, 4, 5])
	})
	it('infers twice', async () => {
		const kernel = webGpGpu
			.bind(inference({ custom: [undefined, undefined] }))
			.input({
				a: f32.array('threads.x'),
			})
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x] * 2.0;')
		async function test(a: number[], o: number[]) {
			const { output } = await kernel({ a })
			expect(output).to.deepArrayEqual(o)
		}
		await test([1, 2], [2, 4])
		await test([1, 2, 3], [2, 4, 6])
	})
})
describe('diverse', () => {
	it('defines', async () => {
		const kernel = webGpGpu
			.bind(inputs({ a: f32.array('threads.x') }))
			.common({ b: f32.array('threads.x').value([2, 4, 6]) })
			.define({ declarations: 'fn myFunc(a: f32, b: f32) -> f32 { return a + b; }' })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = myFunc(a[thread.x], b[thread.x]);')
		const { output } = await kernel({ a: Float32Array.from([1, 2, 3]).buffer })
		expect(output).to.deepArrayEqual([3, 6, 9])
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
		expect(output.length).to.equal(length)
	})
	it('expand workGroupSize', async () => {
		const kernel = webGpGpu
			.input({ a: vec2f.array('threads.x'), b: vec2f.array('threads.x') })
			.output({ output: vec2f.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x]+b[thread.x];')
		expect(kernel.toString()).to.match(/@compute @workgroup_size\(\d{2,},/)
	})
	it('strides', async () => {
		const kernel = webGpGpu
			.common({ b: f32.array('threads.y').value([4, 5]) })
			.input({ a: f32.array('threads.x') })
			.output({
				outputA: f32.array('threads.x', 'threads.y'),
				outputB: f32.array('threads.y', 'threads.x'),
			})
			.kernel(/*wgsl*/ `
		outputA[dot(thread.xy, outputAStride)] = a[thread.x]+b[thread.y];
		outputB[dot(thread.yx, outputBStride)] = a[thread.x]+b[thread.y];
		`)
		expect(kernel.toString()).to.match(/const outputAStride = vec2u\(/)
		expect(kernel.toString()).to.match(/var<private> outputBStride: vec2u;/)
		const { outputA, outputB } = await kernel({ a: [1, 2, 3] })
		expect(outputA).to.deepArrayEqual([
			[5, 6],
			[6, 7],
			[7, 8],
		])
		expect(outputB).to.deepArrayEqual([
			[5, 6, 7],
			[6, 7, 8],
		])
	})
	it('concurs', async () => {
		const kernel = webGpGpu
			.input({ a: f32.array('threads.x') })
			.output({ output: f32.array('threads.x') })
			.kernel('output[thread.x] = a[thread.x] * 2.0;')
		const [r1, r2] = await Promise.all([kernel({ a: [1, 2] }), kernel({ a: [3, 4] })])
		expect(r1.output).to.deepArrayEqual([2, 4])
		expect(r2.output).to.deepArrayEqual([6, 8])
	})
})
