import { type Buffable, isBuffable } from '../buffable/buffable'
import { mapEntries } from '../hacks'
import { type AnyInference, type DeducedInference, resolvedSize } from '../inference'
import { ParameterError } from '../types'
import type { InputType } from '../webgpgpu'
import { Bindings, type WgslEntry } from './bindings'
import { inputGroupEntry, layoutGroupEntry, wgslEntries } from './io'

export class InputBindings<
	Inferences extends AnyInference,
	InputSpecs extends Record<string, Buffable>,
> extends Bindings<Inferences & DeducedInference<InputSpecs[keyof InputSpecs]>> {
	public readonly wgslEntries: Record<string, WgslEntry>
	private readonly inputSpecs: { name: string; buffable: Buffable<AnyInference> }[]
	constructor(inputSpecs: InputSpecs) {
		super()
		// TODO default values
		this.wgslEntries = wgslEntries(inputSpecs)
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
				resolvedSize(buffable.sizes, inferences),
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
