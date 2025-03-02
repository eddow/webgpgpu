import type { Buffable } from 'src/buffable/buffable'
import { ParameterError } from '../types'
import type { GPUUnboundGroupLayoutEntry } from './bindings'
// TODO: dynamically create UBOs with values of fixed size (when `#define` is usable: code parsing)
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
		default: {
			return {
				layoutEntry: {
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
				},
				// TODO: if < 16Kb & read-only & fixed-size, var<uniform>
				// cf. device.limits.maxUniformBuffersPerShaderStage
				// cf. device.limits.maxUniformBufferBindingSize
				declaration: `var<storage, ${readOnly ? 'read' : 'read_write'}> ${name} : array<${buffable.wgslSpecification}>;`,
			}
		}
	}
}

export function inputGroupEntry(
	device: GPUDevice,
	name: string,
	size: number[],
	data: ArrayBuffer
): GPUBindingResource {
	switch (size.length) {
		case 0: {
			const buffer = device.createBuffer({
				label: name,
				size: data.byteLength,
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
			throw new Error(`Not implemented (IGE dimension ${size.length})`)
	}
}

export interface OutputEntryDescription {
	name: string
	resource: GPUBindingResource
	encoder(commandEncoder: GPUCommandEncoder): void
	read(): Promise<ArrayBuffer>
}

export function outputGroupEntry(
	device: GPUDevice,
	name: string,
	size: number[],
	elementByteSize: number
): OutputEntryDescription {
	const totalSize = size.reduce((a, b) => a * b, elementByteSize)
	// TODO: optimize w/ textures: switch (size.length) {
	const outputBuffer = device.createBuffer({
		label: `${name}-output-buffer`,
		size: totalSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, // Used in compute shader
	})
	const readBuffer = device.createBuffer({
		label: `${name}-read-buffer`,
		size: totalSize,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, // Used for CPU read back
	})
	function encoder(commandEncoder: GPUCommandEncoder) {
		commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, totalSize)
	}
	async function read() {
		await readBuffer.mapAsync(GPUMapMode.READ)
		// `getMappedRange` returns a view in the GPU memory - it has to be copied then unmapped (to free GPU memory)
		try {
			return readBuffer.getMappedRange().slice(0)
		} finally {
			readBuffer.unmap()
		}
	}
	return { resource: { buffer: outputBuffer }, encoder, read, name }
}
