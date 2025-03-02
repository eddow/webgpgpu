import { mapEntries } from '../hacks'
import {
	type AnyInference,
	type CreatedInferences,
	type Inferred,
	extractInference,
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

	constructor(private inferred: Input) {
		super()
		this.wgslEntries = mapEntries(inferred, () => ({
			size: [],
		}))
		this.addedInferences = infer({}, inferred)
	}
	private get dimensioned() {
		return Object.entries(this.inferred).map(([name, value]) => ({
			name,
			dimension: (Array.isArray(value) ? value.length : 1) as 1 | 2 | 3 | 4,
		}))
	}
	init() {
		function typeName(dimension: 1 | 2 | 3 | 4) {
			const type = [undefined, 'u32', 'vec2u', 'vec3u', 'vec4u'][dimension]
			if (type === undefined) throw new Error(`Invalid inferred dimension ${dimension}`)
			return type
		}
		return this.dimensioned.map(({ name, dimension }) => ({
			declaration: `var<uniform> ${name} : ${typeName(dimension)};`,
			layoutEntry: {
				visibility: GPUShaderStage.COMPUTE,
				buffer: { type: 'uniform' } as GPUBufferBindingLayout,
			},
		}))
	}
	entries(inferences: AnyInference) {
		const { device } = this
		return this.dimensioned.map(({ name, dimension }) => {
			const buffer = device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			})
			const value = extractInference(inferences, name, dimension, 1)
			device.queue.writeBuffer(buffer, 0, new Uint32Array(value!))
			return {
				resource: { buffer },
			}
		})
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
