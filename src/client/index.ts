import { provideGpu } from '../system'

if (!navigator.gpu) throw new Error('WebGPU not supported by browser')
provideGpu(() => navigator.gpu!)

export * from '..'
