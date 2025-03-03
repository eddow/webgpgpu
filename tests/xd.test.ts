import { expect } from 'chai'
import createWebGpGpu, { type RootWebGpGpu, resolvedSize } from 'webgpgpu'
import { XbYTests, multiplication2 } from './xd.data'

let webGpGpu: RootWebGpGpu

before(async () => {
	webGpGpu = await createWebGpGpu()
})
after(() => {
	webGpGpu.dispose()
})
describe('xd-arrays', () => {
	// multiply two 1D-arrays to create a matrix of all products
	describe('multiply 2D', () => {
		for (const m in multiplication2) {
			const { a, b, r } = multiplication2[m]
			it(m, async () => {
				const kernel = webGpGpu
					//@ts-expect-error Inference validation: we mix everything here
					.common({ a, b })
					//@ts-expect-error
					.output({ m: r.buffable })
					.kernel(/*wgsl*/ `
		let stride = vec2u(threads.y, 1);
		m[dot(thread.xy, stride)] = a[thread.x]*b[thread.y];
		`)
				const { m } = await kernel({})
				expect(m).to.deepArrayEqual(r.value)
			})
		}
	})

	// Write an input array in an ArrayBuffer and read the same from the buffer
	describe('X->buffer->Y: X~Y', () => {
		for (const [name, { buffable, value }] of Object.entries(XbYTests))
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
