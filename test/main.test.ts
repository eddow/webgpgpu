import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { WebGpGpu, f32 } from '../src/server'

afterAll(() => {
	WebGpGpu.dispose()
})
describe('description', () => {
	test('todo', async () => {
		const webGpGpu = await WebGpGpu.root
		const spec = webGpGpu!
			.commonArray('a', f32, [1, 2, 3], 3)
			.commonUniform('b', f32, 2)
			.createFunction(
				[3],
				/*wgsl*/ `
	outputBuffer[thread.x] = a[thread.x] * b;
`
			)
		const rv = await spec([3])
		expect(rv).toBeDefined()
		expect(rv).toHaveLength(3)
		expect(rv[0]).toBe(2)
		expect(rv[1]).toBe(4)
		expect(rv[2]).toBe(6)
	})
})
