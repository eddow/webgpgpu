import { expect } from 'chai'
import createWebGpGpu, { type RootWebGpGpu, f32, type RootGGBatch, GGBatch } from 'webgpgpu'

let webGpGpu: RootWebGpGpu
let ggBatch: RootGGBatch

before(async () => {
	webGpGpu = await createWebGpGpu()
	ggBatch = GGBatch.createRoot(webGpGpu)
})
after(() => {
	webGpGpu.dispose()
})
describe('batch', () => {
	it('batches', async () => {
		const batch = ggBatch
			.common({ c: f32.array('threads.y').value([1, 2]) })
			.input({ i: f32.array('threads.x') })
			.output({ o: f32.array('threads.y', 'threads.x') })
			.batch(/*wgsl*/ `
		o[dot(thread.zyx, oStride)] = i[dot(thread.zx, iStride)]*c[thread.y];
		`)
		const kernel = batch()
		async function aTest(i: number[]) {
			const { o } = await kernel({ i })
			expect(o).to.deepArrayEqual([i, i.map((v) => v * 2)])
		}
		await Promise.all([aTest([1, 2, 3]), aTest([5, 7, 9]), aTest([-7, -11, -13])])
	})
	// TODO:tests: edge-cases & failures
})
