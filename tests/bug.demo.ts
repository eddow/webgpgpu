import { expect } from 'chai'
import { after, before, describe, it } from 'mocha'

import { create, globals } from 'webgpu'

Object.assign(globalThis, globals)
function createBug() {
	return BugCreate.createRoot(create([]))
}

class BugCreate {
	static async createRoot(gpu: GPU): Promise<BugCreate> {
		const adapter = await gpu.requestAdapter()
		const device = await adapter!.requestDevice()
		return new BugCreate(device)
	}
	private constructor(public device?: GPUDevice) {}
	dispose() {
		if (this.device) this.device.destroy()
		this.device = undefined
	}
}

let bugCreate: BugCreate | undefined

before(async () => {
	bugCreate = await createBug()
})
after(() => {
	console.log('dispose')
	bugCreate?.dispose()
})
describe('bug', () => {
	it('bugs', async () => {
		await new Promise((resolve) => setTimeout(resolve, 200))
		expect('bug').to.exist
	})
})
