import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import createWebGpGpu, { Vector2, type WebGpGpu, f32, threads, vec2f } from '../src/server'
import { ArraySizeValidationError } from '../src/typedArrays'

let webGpGpu: WebGpGpu

beforeAll(async () => {
	webGpGpu = await createWebGpGpu()
})
afterAll(() => {
	webGpGpu.dispose()
})
describe('overall', () => {
	test('toString', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu.workGroup(42).kernel('')
		expect(kernel).toBeDefined()
		expect(kernel).toBeInstanceOf(Function)
		expect(kernel.toString()).toMatch('@group(0) @binding(0) var<uniform> threads : vec3u;')
		expect(kernel.toString()).toMatch(`
@compute @workgroup_size(42,1,1)
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
		
	}
}
`)
	})
	test('no argument', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu.output({ output: f32.array(threads.x) }).kernel('output[0] = 42;')
		const { output } = await kernel({})
		expect(output).toBeDefined()
		expect(output.toArray()).toMatchObject([42])
	})
})

describe('inputs', () => {
	test('a uniform - TypedArray', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32 })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = a;')
		const { output } = await kernel({ a: Float32Array.from([43]) }, { x: 3 })
		expect(output).toBeDefined()
		expect(output.toArray()).toMatchObject([43, 43, 43])
	})
	test('a uniform - value', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32 })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = a;')
		const { output } = await kernel({ a: 44 }, { x: 3 })
		expect(output.toArray()).toMatchObject([44, 44, 44])
	})
	test('an float array - TypedArray', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3) })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = a[thread.x]*3;')
		const { output } = await kernel({ a: Float32Array.from([11, 12, 13]) }, { x: 3 })
		expect(output.toArray()).toMatchObject([33, 36, 39])
	})
	test('an float array - value', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3) })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = a[thread.x]*3;')
		const { output } = await kernel({ a: [11, 12, 13] }, { x: 3 })
		expect(output.toArray()).toMatchObject([33, 36, 39])
	})
	test('an vec2 array - TypedArray', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: vec2f.array(3) })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const { output } = await kernel({ a: Float32Array.from([1, 11, 2, 12, 3, 13]) }, { x: 3 })
		expect(output.toArray()).toMatchObject([36, 42, 48])
	})
	test('an vec2 array - value', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: vec2f.array(3) })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const { output } = await kernel(
			{
				a: [
					[1, 11],
					[2, 12],
					[3, 13],
				],
			},
			{ x: 3 }
		)
		expect(output.toArray()).toMatchObject([36, 42, 48])
	})
	test('an vec2 array - transform', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: Vector2.array(3) })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const { output } = await kernel(
			{
				a: [
					{ x: 1, y: 11 },
					{ x: 2, y: 12 },
					{ x: 3, y: 13 },
				],
			},
			{ x: 3 }
		)
		expect(output.toArray()).toMatchObject([36, 42, 48])
	})
	test('two arguments', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3), b: f32 })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = a[thread.x] * b;')
		const { output } = await kernel(
			{
				a: Float32Array.from([1, 2, 3]),
				b: 5,
			},
			{ x: 3 }
		)
		expect(output.toArray()).toMatchObject([5, 10, 15])
	})
})
describe('size inference', () => {
	test('infer', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.common({
				a: f32.array(threads.x).value([1, 2, 3]),
				b: f32.array(threads.y).value([4, 5, 6, 7, 8]),
			})
			.kernel('')
		expect(kernel.toString()).toMatch('@workgroup_size(4,8,') // The last one depends on the hardware configuration
	})
	test('assert', async () => {
		//const webGpGpu = await webGpGpuPromise!
		expect(() =>
			webGpGpu.common({
				a: f32.array(threads.x).value([1, 2, 3]),
				b: f32.array(threads.x).value([4, 5, 6, 7, 8]),
			})
		).toThrowError(ArraySizeValidationError)
	})
	test('effect - common', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.common({ a: f32.array(threads.x).value([1, 2, 3]) })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = a[thread.x]+3.;')

		const { output } = await kernel({})
		expect(output.toArray()).toMatchObject([4, 5, 6])
	})
	test('effect - input', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.input({ a: f32.array(threads.x) })
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = a[thread.x]+2.;')
		const { output } = await kernel({ a: [1, 2, 3] })
		expect(output.toArray()).toMatchObject([3, 4, 5])
	})
	// TODO: check TypedArray sizes & infer ?
})
describe('diverse', () => {
	test('defined', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.input({ a: f32.array(threads.x) })
			.common({ b: f32.array(threads.x).value([2, 4, 6]) })
			.defined('fn myFunc(a: f32, b: f32) -> f32 { return a + b; }')
			.output({ output: f32.array(threads.x) })
			.kernel('output[thread.x] = myFunc(a[thread.x], b[thread.x]);')
		const { output } = await kernel({ a: Float32Array.from([1, 2, 3]) })
		expect(output.toArray()).toMatchObject([3, 6, 9])
	})
	test('name conflict', async () => {
		//const webGpGpu = await webGpGpuPromise!
		const kernel = webGpGpu
			.input({ a: f32.array(threads.x) })
			.common({ b: f32.array(threads.x).value([2, 4, 6]) })
		expect(() => kernel.input({ a: f32.array(threads.y) })).toThrowError()
		expect(() => kernel.input({ b: f32.array(threads.y) })).toThrowError()
	})
})
describe('outputs', () => {
	test('todo', () => {
		expect(true).toBe(true)
	})
})
describe('half-size convert', () => {
	// .array(...).transform(...)
	test('todo', () => {
		expect(true).toBe(true)
	})
})
