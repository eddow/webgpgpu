import { type AnyInference, type DeducedInference, type SizeSpec, resolvedSize } from '../inference'
import { ParameterError } from '../types'
import type { Buffable } from '../types/buffable'
import { isBuffable } from '../types/ggData'
import type { InputType } from '../webgpgpu'
import { Bindings } from './bindings'
import { inputGroupEntry, layoutGroupEntry } from './io'

export class InputBindings<
	Inferences extends AnyInference,
	InputSpecs extends Record<string, Buffable>,
> extends Bindings<Inferences & DeducedInference<InputSpecs[keyof InputSpecs]>> {
	public readonly wgslNames: string[]
	private readonly inputSpecs: { name: string; buffable: Buffable<AnyInference> }[]
	constructor(inputSpecs: InputSpecs) {
		super()
		// TODO default values
		this.wgslNames = Object.keys(inputSpecs)
		this.inputSpecs = Object.entries(inputSpecs).map(([name, buffable]) => {
			if (!isBuffable(buffable)) throw new ParameterError(`Bad value for input \`${name}\``)
			return {
				name,
				buffable: buffable as Buffable<AnyInference>,
			}
		})
	}

	init() {
		return this.inputSpecs.map(({ name, buffable }) => layoutGroupEntry(name, buffable, true))
	}
	entries(
		inferences: Inferences,
		inputs: { [K in keyof InputSpecs]: InputType<InputSpecs[K]> },
		reasons: Record<string, string>
	) {
		const { device, inputSpecs } = this
		return inputSpecs.map(({ name, buffable }) => {
			const arrayBuffer = buffable.toArrayBuffer(
				inputs[name],
				inferences,
				`Given input ${name}`,
				reasons
			)
			const resource = inputGroupEntry(
				device,
				name,
				resolvedSize(buffable.size, inferences),
				arrayBuffer
			)
			return { resource }
		})
	}
}

export default function inputs<
	Inferences extends AnyInference,
	InputSpecs extends Record<string, Buffable<Inferences>>,
>(inputSpecs: InputSpecs) {
	return new InputBindings<Inferences, InputSpecs>(inputSpecs)
}
