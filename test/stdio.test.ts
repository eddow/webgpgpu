import { afterAll, describe, expect, test } from 'vitest'
import { b } from 'vitest/dist/chunks/suite.qtkXWc6R.js'
import { Vector2, WebGpGpu, f32, threads, vec2f } from '../src/server'
import { ArraySizeValidationError } from '../src/typedArrays'

afterAll(() => {
	WebGpGpu.dispose()
})
describe('overall', () => {
	test('toString', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu.workGroup(42).kernel('')
		expect(spec).toBeDefined()
		expect(spec).toBeInstanceOf(Function)
		expect(spec.toString()).toMatch('@group(0) @binding(0) var<uniform> threads : vec3u;')
		expect(spec.toString()).toMatch(`
@compute @workgroup_size(42,1,1)
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
		
	}
}
`)
	})
	test('no argument', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu.workGroup(3).kernel('outputBuffer[thread.x] = 42;')
		const rv = await spec({}, 3)
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([42, 42, 42]))
	})
})

describe('inputs', () => {
	test('a uniform - TypedArray', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu.workGroup(3).input({ a: f32 }).kernel('outputBuffer[thread.x] = a;')
		const rv = await spec({ a: Float32Array.from([43]) }, 3)
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([43, 43, 43]))
	})
	test('a uniform - value', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu.workGroup(3).input({ a: f32 }).kernel('outputBuffer[thread.x] = a;')
		const rv = await spec({ a: 44 }, 3)
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([44, 44, 44]))
	})
	test('an float array - TypedArray', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3) })
			.kernel('outputBuffer[thread.x] = a[thread.x]*3;')
		const rv = await spec({ a: Float32Array.from([11, 12, 13]) }, 3)
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([33, 36, 39]))
	})
	test('an float array - value', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3) })
			.kernel('outputBuffer[thread.x] = a[thread.x]*3;')
		const rv = await spec({ a: [11, 12, 13] }, 3)
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([33, 36, 39]))
	})
	test('an vec2 array - TypedArray', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu
			.workGroup(3)
			.input({ a: vec2f.array(3) })
			.kernel('outputBuffer[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const rv = await spec({ a: Float32Array.from([1, 11, 2, 12, 3, 13]) }, 3)
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([36, 42, 48]))
	})
	test('an vec2 array - value', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu
			.workGroup(3)
			.input({ a: vec2f.array(3) })
			.kernel('outputBuffer[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const rv = await spec(
			{
				a: [
					[1, 11],
					[2, 12],
					[3, 13],
				],
			},
			3
		)
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([36, 42, 48]))
	})
	test('an vec2 array - transform', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu
			.workGroup(3)
			.input({ a: Vector2.array(3) })
			.kernel('outputBuffer[thread.x] = (a[thread.x]*3).x + (a[thread.x]*3).y;')
		const rv = await spec(
			{
				a: [
					{ x: 1, y: 11 },
					{ x: 2, y: 12 },
					{ x: 3, y: 13 },
				],
			},
			3
		)
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([36, 42, 48]))
	})
	test('two arguments', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu
			.workGroup(3)
			.input({ a: f32.array(3), b: f32 })
			.kernel('outputBuffer[thread.x] = a[thread.x] * b;')
		const rv = await spec(
			{
				a: Float32Array.from([1, 2, 3]),
				b: 5,
			},
			3
		)
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([5, 10, 15]))
	})
})
describe('size inference', () => {
	test('infer', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu
			.input({ a: f32.array(threads.x), b: f32.array(threads.y) })
			.common({ a: [1, 2, 3], b: [4, 5, 6, 7, 8] })
			.kernel('')
		expect(spec.toString()).toMatch('@workgroup_size(4,8,') // The last one depends on the hardware configuration
	})
	test('assert', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu.input({ a: f32.array(threads.x), b: f32.array(threads.x) })
		expect(() => spec.common({ a: [1, 2, 3], b: [4, 5, 6, 7, 8] })).toThrowError(
			ArraySizeValidationError
		)
	})
	test('effect - common', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu
			.input({ a: f32.array(threads.x) })
			.common({ a: [1, 2, 3] })
			.kernel('outputBuffer[thread.x] = a[thread.x]+3.;')

		const rv = await spec({})
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([4, 5, 6]))
	})
	test('effect - input', async () => {
		const webGpGpu = (await WebGpGpu.root) as WebGpGpu
		const spec = webGpGpu
			.input({ a: f32.array(threads.x) })
			.kernel('outputBuffer[thread.x] = a[thread.x]+2.;')
		const rv = await spec({ a: [1, 2, 3] })
		expect(rv).toBeDefined()
		expect(rv).toMatchObject(Float32Array.from([3, 4, 5]))
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
