import { type AnyInference, type DeducedInference, resolvedSize } from '../inference'
import { type InputXD, ParameterError } from '../types'
import type { Buffable, ValuedBuffable } from '../types/buffable'
import { isBuffable } from '../types/ggData'
import { Bindings, type GPUUnboundGroupEntry } from './bindings'
import { inputGroupEntry, layoutGroupEntry } from './io'

export class CommonBindings<
	Inferences extends AnyInference,
	CommonSpecs extends Record<string, ValuedBuffable<Inferences>>,
> extends Bindings<Inferences & DeducedInference<CommonSpecs[keyof CommonSpecs]['buffable']>> {
	public readonly wgslNames: string[]
	commonSpecs: { name: string; buffable: Buffable<Inferences>; value: InputXD }[]
	private precomputedEntries?: GPUUnboundGroupEntry[]
	constructor(commonSpecs: CommonSpecs) {
		super()
		this.wgslNames = Object.keys(commonSpecs)
		this.commonSpecs = Object.entries(commonSpecs).map(([name, { buffable, value }]) => {
			if (!isBuffable(buffable)) throw new ParameterError(`Bad value for input \`${name}\``)
			return {
				name,
				buffable,
				value,
			}
		})
	}
	init(inferences: Inferences, reasons: Record<string, string>) {
		const { device, commonSpecs } = this
		this.precomputedEntries = commonSpecs.map(({ name, buffable, value }) => {
			const typeArray = buffable.toArrayBuffer(value, inferences, `Given input ${name}`, reasons)
			const resource = inputGroupEntry(
				device,
				name,
				resolvedSize(buffable.size, inferences),
				typeArray
			)
			return { resource }
		})
		return commonSpecs.map(({ name, buffable }) => layoutGroupEntry(name, buffable, true))
	}
	entries() {
		return this.precomputedEntries!
	}
}

export default function commons<
	Inferences extends AnyInference,
	CommonSpecs extends Record<string, ValuedBuffable<Inferences>>,
>(commonSpecs: CommonSpecs) {
	return new CommonBindings<Inferences, CommonSpecs>(commonSpecs)
}
