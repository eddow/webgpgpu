import { create, globals } from 'webgpu'

Object.assign(globalThis, globals)
export function createBug() {
	return BugCreate.createRoot(create([]))
}

export class BugCreate {
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

async function main() {
	bugCreate = await createBug()
	await new Promise((resolve) => setTimeout(resolve, 2000))
	bugCreate.dispose()
}

main()
