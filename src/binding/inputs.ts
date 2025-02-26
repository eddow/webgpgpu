import { isBuffable } from '../buffable'
import { Buffable } from '../buffers'
import { type AnyInference, resolvedSize } from '../inference'
import { ParameterError } from '../types'
import type { InputType } from '../webgpgpu'
import { Bindings } from './bindings'
import { inputGroupEntry, layoutGroupEntry } from './io'

export class InputBindings<InputSpecs extends Record<string, Buffable>> extends Bindings {
	public readonly wgslNames: string[]
	private readonly inputSpecs: { name: string; buffable: Buffable<AnyInference> }[]
	constructor(inputSpecs: InputSpecs) {
		super()
		// TODO default values
		this.wgslNames = Object.keys(inputSpecs)
		for (const name in inputSpecs)
			if (!isBuffable(inputSpecs[name])) throw new ParameterError(`Bad value for input \`${name}\``)
		this.inputSpecs = Object.entries(inputSpecs).map(([name, buffable]) => ({
			name,
			buffable: buffable as Buffable<AnyInference>,
		}))
	}
	init() {
		return this.inputSpecs.map(({ name, buffable }) => layoutGroupEntry(name, buffable, true))
	}
	entries(
		inferences: AnyInference,
		inputs: Record<keyof InputSpecs, InputType<InputSpecs[keyof InputSpecs]>>
	) {
		const { device, inputSpecs } = this
		return inputSpecs.map(({ name, buffable }) => {
			const typeArray = buffable.toTypedArray(inferences, inputs[name], name, {})
			const resource = inputGroupEntry(
				device,
				name,
				resolvedSize(buffable.size, inferences),
				typeArray
			)
			return { resource }
		})
	}
}

export default function inputs<InputSpecs extends Record<string, Buffable>>(
	inputSpecs: InputSpecs
) {
	return new InputBindings(inputSpecs)
}
