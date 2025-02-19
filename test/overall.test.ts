import { afterAll, describe, expect, test } from 'vitest'
import { WebGpGpu, f32 } from '../src/server'

afterAll(() => {
	WebGpGpu.dispose()
})
describe('overall tests', () => {
	test('basic string', async () => {
		const webGpGpu = await WebGpGpu.root
		const spec = webGpGpu!.kernel([42], '')
		expect(spec).toBeDefined()
		expect(spec).toBeInstanceOf(Function)
		expect(spec.toString()).toMatch('@group(0) @binding(0) var<uniform> threads : vec3u;')
		expect(spec.toString()).toMatch(`
@compute @workgroup_size(42)
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
		
	}
}
`)
	})

	test('basic work', async () => {
		const webGpGpu = await WebGpGpu.root
		const spec = webGpGpu!
			.commonArray('a', f32, [1, 2, 3], 3)
			.commonUniform('b', f32, 2)
			.kernel(
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
	test('class array', async () => {
		class TA extends Array<number> {
			constructor(items: number | number[]) {
				super(typeof items === 'number' ? items : 0)
				if (Array.isArray(items)) this.push(...items)
			}
			get x() {
				return this[0]
			}
			get y() {
				return this[1]
			}
		}
		const x = new TA([1, 2])
		expect(x.x).toBe(1)
	})
})
