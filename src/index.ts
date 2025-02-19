import { system } from './adapter'
if (!system.getGpu) throw new Error('Wrongly linked library usage') // client/server index makes the link
export * from './helloWorld'
export * from './workGroup'
export * from './dataTypesList'
export * from './webgpgpu'
