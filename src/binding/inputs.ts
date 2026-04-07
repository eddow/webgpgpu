import { type Buffable, type IBuffable, isBuffable } from '../buffable/buffable'
import { Struct } from '../buffable/struct'
import { type AnyInference, type DeducedInference, resolvedSize } from '../inference'
import { type InputType, ParameterError } from '../types'
import { Bindings, type WgslEntry } from './bindings'
import { inputGroupEntry, layoutGroupEntry, wgslEntries } from './io'

interface InputSpec {
	name: string
	buffable: IBuffable<AnyInference>
}

export class InputBindings<
	Inferences extends AnyInference,
	InputSpecs extends Record<string, IBuffable>,
> extends Bindings<Inferences & DeducedInference<InputSpecs[keyof InputSpecs]>> {
	public readonly wgslEntries: Record<string, WgslEntry>
	private readonly tierA: InputSpec[]
	private readonly tierB: InputSpec[]
	private packedStruct?: Struct<AnyInference, Record<string, Buffable<AnyInference, any, []>>>
	private readonly dispatchBuffers = new WeakMap<{}, GPUBuffer[]>()

	constructor(inputSpecs: InputSpecs) {
		super()
		this.wgslEntries = wgslEntries(inputSpecs)
		const all: InputSpec[] = Object.entries(inputSpecs).map(([name, buffable]) => {
			if (!isBuffable(buffable)) throw new ParameterError(`Bad value for input \`${name}\``)
			return { name, buffable: buffable as IBuffable<AnyInference> }
		})
		this.tierA = all.filter(({ buffable }) => buffable.sizes.length === 0)
		this.tierB = all.filter(({ buffable }) => buffable.sizes.length > 0)
		if (this.tierA.length > 0) {
			const descriptor = Object.fromEntries(
				this.tierA.map(({ name, buffable }) => [
					name,
					buffable.base as Buffable<AnyInference, any, []>,
				])
			)
			this.packedStruct = new Struct('_Params', descriptor)
		}
	}

	init() {
		const entryDescriptors = []
		let definitions: string | undefined
		let preamble: string | undefined

		if (this.packedStruct) {
			const isUniform =
				this.packedStruct.bytesPerAtomic <= this.device.limits.maxUniformBufferBindingSize
			const addressSpace = isUniform ? 'uniform' : 'storage, read'
			const bufferType: GPUBufferBindingType = isUniform ? 'uniform' : 'read-only-storage'
			entryDescriptors.push({
				declaration: `var<${addressSpace}> _params : _Params;`,
				layoutEntry: {
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: bufferType } as GPUBufferBindingLayout,
				},
			})
			definitions = this.packedStruct.wgsl
			preamble = this.tierA.map(({ name }) => `let ${name} = _params.${name};`).join('\n\t\t')
		}

		for (const { name, buffable } of this.tierB)
			entryDescriptors.push(layoutGroupEntry(name, buffable, true))

		return { entryDescriptors, definitions, preamble }
	}

	entries(
		inputs: { [K in keyof InputSpecs]: InputType<InputSpecs[K]> },
		inferences: Inferences,
		reasons: Record<string, string>
	) {
		const { device } = this
		const result = []
		const buffers: GPUBuffer[] = []

		if (this.packedStruct) {
			const packed = this.packTierA(inputs, inferences, reasons)
			const isUniform =
				this.packedStruct.bytesPerAtomic <= device.limits.maxUniformBufferBindingSize
			const usage = isUniform
				? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
				: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			const buffer = device.createBuffer({
				label: '_params',
				size: packed.byteLength,
				usage,
			})
			device.queue.writeBuffer(buffer, 0, packed)
			buffers.push(buffer)
			result.push({ resource: { buffer } })
		}

		for (const { name, buffable } of this.tierB) {
			const arrayBuffer = buffable.toArrayBuffer(
				inputs[name],
				inferences,
				`Given input ${name}`,
				reasons
			)
			const resource = inputGroupEntry(
				device,
				name,
				resolvedSize(buffable.sizes, inferences),
				arrayBuffer
			)
			buffers.push((resource as GPUBufferBinding).buffer)
			result.push({ resource })
		}

		this.dispatchBuffers.set(inputs, buffers)
		return result
	}

	dispose(inputs: {}) {
		const buffers = this.dispatchBuffers.get(inputs)
		if (buffers) {
			for (const buf of buffers) buf.destroy()
			this.dispatchBuffers.delete(inputs)
		}
	}

	// Byte-level packing rather than Struct.toArrayBuffer: each field is serialized through its
	// own buffable.toArrayBuffer (handles TypedArray / ArrayBuffer / plain value uniformly),
	// then placed at the struct's paddedDescriptor offsets. Layout stays in sync via the struct.
	private packTierA(
		inputs: { [K in keyof InputSpecs]: InputType<InputSpecs[K]> },
		inferences: Inferences,
		reasons: Record<string, string>
	): ArrayBuffer {
		const struct = this.packedStruct!
		const packed = new ArrayBuffer(struct.bytesPerAtomic)
		const target = new Uint8Array(packed)
		for (const { name, offset } of struct.paddedDescriptor) {
			const spec = this.tierA.find((s) => s.name === name)!
			const fieldBytes = spec.buffable.toArrayBuffer(
				inputs[name],
				inferences,
				`Given input ${name}`,
				reasons
			)
			target.set(new Uint8Array(fieldBytes), offset)
		}
		return packed
	}
}

export default function inputs<
	Inferences extends AnyInference,
	InputSpecs extends Record<string, IBuffable<Inferences>>,
>(inputSpecs: InputSpecs) {
	return new InputBindings<Inferences, InputSpecs>(inputSpecs)
}
