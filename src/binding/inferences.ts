import { mapEntries } from '../hacks'
import {
	type AnyInference,
	type CreatedInferences,
	extractInference,
	type Inferred,
	infer,
} from '../inference'
import { Bindings, type WgslEntry } from './bindings'

// TODO: Type-check no interference in inference (no common key)
export class InferenceBindings<
	Inferences extends AnyInference,
	Input extends Record<
		string,
		| Inferred
		| readonly [Inferred, Inferred]
		| readonly [Inferred, Inferred, Inferred]
		| readonly [Inferred, Inferred, Inferred, Inferred]
	>,
> extends Bindings<Inferences> {
	public readonly wgslEntries: Record<string, WgslEntry>
	public readonly addedInferences: CreatedInferences<Input>
	private readonly dispatchBuffers = new WeakMap<{}, GPUBuffer[]>()

	constructor(private inferred: Input) {
		super()
		this.wgslEntries = mapEntries(inferred, () => ({
			sizes: [],
		}))
		this.addedInferences = infer({}, inferred)
	}
	private get dimensioned() {
		return Object.entries(this.inferred).map(([name, value]) => ({
			name,
			dimension: (Array.isArray(value) ? value.length : 1) as 1 | 2 | 3 | 4,
		}))
	}
	init(_inferences: Inferences, _reasons: Record<string, string>) {
		// TODO: manage constants
		function typeName(dimension: 1 | 2 | 3 | 4) {
			const type = [undefined, 'u32', 'vec2u', 'vec3u', 'vec4u'][dimension]
			if (type === undefined) throw new Error(`Invalid inferred dimension ${dimension}`)
			return type
		}
		return {
			entryDescriptors: this.dimensioned.map(({ name, dimension }) => ({
				declaration: `var<uniform> ${name} : ${typeName(dimension)};`,
				layoutEntry: {
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform' } as GPUBufferBindingLayout,
				},
			})),
		}
	}
	entries(inputs: {}, inferences: AnyInference) {
		const { device } = this
		const buffers: GPUBuffer[] = []
		const entries = this.dimensioned.map(({ name, dimension }) => {
			const buffer = device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			})
			const value = extractInference(inferences, name, dimension, 1)
			device.queue.writeBuffer(buffer, 0, new Uint32Array(value!))
			buffers.push(buffer)
			return { resource: { buffer } }
		})
		this.dispatchBuffers.set(inputs, buffers)
		return entries
	}
	dispose(inputs: {}) {
		const buffers = this.dispatchBuffers.get(inputs)
		if (buffers) {
			for (const buf of buffers) buf.destroy()
			this.dispatchBuffers.delete(inputs)
		}
	}
}

export default function inferences<
	Inferences extends AnyInference,
	Input extends Record<
		string,
		| Inferred
		| readonly [Inferred, Inferred]
		| readonly [Inferred, Inferred, Inferred]
		| readonly [Inferred, Inferred, Inferred, Inferred]
	>,
>(input: Input) {
	return new InferenceBindings<Inferences, Input>(input)
}
