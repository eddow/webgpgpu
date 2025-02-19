import { initWebGPU } from '../src/server'

describe('description', () => {
	test('todo', async () => {
		await initWebGPU()
		expect(1).toBe(1)
	})
})
