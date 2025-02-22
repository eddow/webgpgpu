import { expect } from 'chai'
import { after, before, describe, it } from 'mocha'
import { create } from 'webgpu'
import { type BugCreate, createBug } from './bug'

let gpu: GPU
let adapter: GPUAdapter
let device: GPUDevice
let bugCreate: BugCreate | undefined

before(async () => {
	//*
	bugCreate = await createBug()
	/*/
	gpu = await create([])
	adapter = (await gpu.requestAdapter())!
	device = await adapter!.requestDevice()
	//*/
})
after(() => {
	console.log('dispose')
	bugCreate?.dispose()
	gpu = null!
	adapter = null!
	device = null!
})
describe('bug', () => {
	it('bugs', async () => {
		await new Promise((resolve) => setTimeout(resolve, 200))
		expect('bug').to.exist
	})
})
