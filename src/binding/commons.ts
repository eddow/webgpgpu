import { type AnyInput, type IBuffable, type ValuedBuffable, isBuffable } from '../buffable'
import { mapEntries } from '../hacks'
import { type AnyInference, type DeducedInference, resolvedSize } from '../inference'
import { ParameterError } from '../types'
import { Bindings, type GPUUnboundGroupEntry, type WgslEntry } from './bindings'
import { inputGroupEntry, layoutGroupEntry, wgslEntries } from './io'

type SubInferences<
	Inferences extends AnyInference,
	CommonSpecs extends Record<string, ValuedBuffable<Inferences>>,
> = Inferences & DeducedInference<CommonSpecs[keyof CommonSpecs]['buffable']>

export class CommonBindings<
	Inferences extends AnyInference,
	CommonSpecs extends Record<string, ValuedBuffable<Inferences>>,
> extends Bindings<SubInferences<Inferences, CommonSpecs>> {
	public readonly wgslEntries: Record<string, WgslEntry<SubInferences<Inferences, CommonSpecs>>>
	commonSpecs: { name: string; buffable: IBuffable<Inferences>; value: AnyInput }[]
	private precomputedEntries?: GPUUnboundGroupEntry[]
	constructor(commonSpecs: CommonSpecs) {
		super()
		this.wgslEntries = wgslEntries(mapEntries(commonSpecs, ({ buffable }) => buffable))
		this.wgslEntries = mapEntries(commonSpecs, ({ buffable: { sizes, elementSizes } }) => ({
			sizes: [...sizes, ...elementSizes],
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
			const typeArray = buffable.toArrayBuffer(value, inferences, `Given common ${name}`, reasons)
			const resource = inputGroupEntry(
				device,
				name,
				resolvedSize(buffable.sizes, inferences),
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
