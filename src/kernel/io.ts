import type { Buffable } from '../buffable'
import { ParameterError, type TypedArray, type TypedArrayConstructor } from '../types'

export const inferredBindGroupIndex = 0
export const commonBindGroupIndex = 1
export const inputBindGroupIndex = 2
export const outputBindGroupIndex = 3

export interface BindingEntryDescription {
	layoutEntry: GPUBindGroupLayoutEntry
	description: string
}

export interface OutputEntryDescription {
	name: string
	resource: GPUBindingResource
	encoder(commandEncoder: GPUCommandEncoder): void
	read(): Promise<TypedArray>
}

export function layoutGroupEntry(
	name: string,
	buffable: Buffable,
	group: number,
	binding: number,
	readOnly: boolean
): BindingEntryDescription {
	// TODO: If size is already inferred, write it here
	switch (buffable.size.length) {
		case 0: {
			if (!readOnly)
				throw new ParameterError('Uniforms can only be read-only (cannot be used as input)')
			return {
				layoutEntry: {
					binding,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform' },
				},
				description: `@group(${group}) @binding(${binding}) var<uniform> ${name} : ${buffable.wgslSpecification};`,
			}
		}
		case 1: {
			return {
				layoutEntry: {
					binding,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
				},
				description: `@group(${group}) @binding(${binding}) var<storage, ${readOnly ? 'read' : 'read_write'}> ${name} : array<${buffable.wgslSpecification}>;`,
			}
		}
		/* TODO: Textures
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
		/* TODO: Textures
		case 2:
		case 3:*/
		default:
			throw new Error('Not implemented')
	}
}

export function outputGroupEntry(
	device: GPUDevice,
	name: string,
	size: number[],
	elementSize: number,
	bufferType: TypedArrayConstructor<TypedArray>
): OutputEntryDescription {
	const totalSize = elementSize * bufferType.BYTES_PER_ELEMENT * size.reduce((a, b) => a * b, 1)
	switch (size.length) {
		//case 0: impossible - throws on layout
		case 1: {
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
				const rv = new bufferType(readBuffer.getMappedRange().slice(0))
				readBuffer.unmap()
				return rv
			}
			return { resource: { buffer: outputBuffer }, encoder, read, name }
		}
		/* TODO: Textures
		case 2:
		case 3:*/
		default:
			throw new Error('Not implemented')
	}
}
