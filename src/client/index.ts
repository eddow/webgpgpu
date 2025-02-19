import { system } from '../adapter'

if (!navigator.gpu) throw new Error('WebGPU not supported by browser')
system.getGpu = () => navigator.gpu!

export * from '..'
