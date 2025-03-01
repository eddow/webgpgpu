import { type AnyInference, type DeducedInference, resolvedSize } from '../inference'
import { type Buffable, type InputXD, type ValuedBuffable, isBuffable } from '../mapped'
import { ParameterError, mapEntries } from '../types'
import { Bindings, type GPUUnboundGroupEntry, type WgslEntry } from './bindings'
import { inputGroupEntry, layoutGroupEntry } from './io'

type SubInferences<
	Inferences extends AnyInference,
	CommonSpecs extends Record<string, ValuedBuffable<Inferences>>,
> = Inferences & DeducedInference<CommonSpecs[keyof CommonSpecs]['buffable']>

export class CommonBindings<
	Inferences extends AnyInference,
	CommonSpecs extends Record<string, ValuedBuffable<Inferences>>,
> extends Bindings<SubInferences<Inferences, CommonSpecs>> {
	public readonly wgslEntries: Record<string, WgslEntry<SubInferences<Inferences, CommonSpecs>>>
	commonSpecs: { name: string; buffable: Buffable<Inferences>; value: InputXD }[]
	private precomputedEntries?: GPUUnboundGroupEntry[]
	constructor(commonSpecs: CommonSpecs) {
		super()
		this.wgslEntries = mapEntries(commonSpecs, ({ buffable: { size, elementSize } }) => ({
			size: [...size, ...elementSize],
		}))
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
