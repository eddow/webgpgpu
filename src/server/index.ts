import { create, globals } from 'webgpu'
export { create, globals }
import { system } from '../adapter'

export const gpuOptions: string[] = []
system.getGpu = () => create(gpuOptions)!
Object.assign(globalThis, globals)
export * from '../index'
