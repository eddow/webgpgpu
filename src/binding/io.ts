import type { IBuffable } from '../buffable/buffable'
import type { BindingEntryDescription, GPUUnboundGroupLayoutEntry, WgslEntry } from './bindings'
// TODO: dynamically create UBOs with values of fixed size (when `#define` is usable: code parsing)
export function layoutGroupEntry(
	name: string,
	buffable: IBuffable,
	readOnly: boolean
): BindingEntryDescription {
	// TODO: If size is already inferred, write it here
	switch (buffable.sizes.length) {
		case 0: {
			return {
				layoutEntry: {
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
				},
				declaration: `var<storage, ${readOnly ? 'read' : 'read_write'}> ${name} : ${buffable.wgslSpecification};`,
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
	// TODO: Manage UBOs automatically and replace in code `value` by `UBO#.value`
	switch (size.length) {
		case 0: {
			const buffer = device.createBuffer({
				label: name,
				size: data.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			})
			device.queue.writeBuffer(buffer, 0, data)
			return { buffer }
		}
		default: {
			// TODO: Indicate size if inferred + allocate UBO space depending on device limitations
			const buffer = device.createBuffer({
				label: name,
				size: data.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			})
			device.queue.writeBuffer(buffer, 0, data)
			return { buffer }
		}
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

export function wgslEntries(buf: Record<string, IBuffable>) {
	const entries: Record<string, WgslEntry> = {}
	for (const key in buf) {
		const { sizes, elementSizes } = buf[key]
		const bufEntry = { sizes: [...sizes, ...elementSizes] }
		entries[key] = bufEntry
		if (bufEntry.sizes.length >= 2) entries[`${key}Stride`] = { sizes: [] }
	}
	return entries
}
