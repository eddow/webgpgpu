import type { Buffable } from '../buffers'
import { ParameterError, type TypedArray } from '../types'
import type { GPUUnboundGroupLayoutEntry } from './bindings'

export interface BindingEntryDescription {
	declaration: string
	layoutEntry: GPUUnboundGroupLayoutEntry
}
export function layoutGroupEntry(
	name: string,
	buffable: Buffable,
	readOnly: boolean
): BindingEntryDescription {
	// TODO: If size is already inferred, write it here
	switch (buffable.size.length) {
		case 0: {
			if (!readOnly)
				throw new ParameterError('Uniforms can only be read-only (cannot be used as input)')
			return {
				layoutEntry: {
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform' },
				},
				declaration: `var<uniform> ${name} : ${buffable.wgslSpecification};`,
			}
		}
		case 1: {
			return {
				layoutEntry: {
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
				},
				declaration: `var<storage, ${readOnly ? 'read' : 'read_write'}> ${name} : array<${buffable.wgslSpecification}>;`,
			}
		}
		/* TODO: 2~3~4D
		case 2:
		case 3:*/
		default:
			throw new Error('Not implemented')
	}
}

export function inputGroupEntry(
	device: GPUDevice,
	name: string,
	size: number[],
	data: TypedArray
): GPUBindingResource {
	switch (size.length) {
		case 0: {
			const buffer = device.createBuffer({
				label: name,
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			})
			device.queue.writeBuffer(buffer, 0, data)
			return { buffer }
		}
		case 1: {
			const buffer = device.createBuffer({
				label: name,
				size: data.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			})
			device.queue.writeBuffer(buffer, 0, data)
			return { buffer }
		}
		/* TODO: 2~3~4D
		case 2:
		case 3:*/
		default:
			throw new Error('Not implemented')
	}
}
