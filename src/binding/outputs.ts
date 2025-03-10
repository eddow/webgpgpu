import { type IBuffable, isBuffable } from '../buffable/buffable'
import { type AnyInference, type DeducedInference, resolvedSize } from '../inference'
import { type OutputType, ParameterError } from '../types'
import { Bindings, type WgslEntry } from './bindings'
import { type OutputEntryDescription, layoutGroupEntry, outputGroupEntry, wgslEntries } from './io'

type Outputs<Specs extends Record<string, IBuffable>> = { [K in keyof Specs]: OutputType<Specs[K]> }
type SubInferences<
	Inferences extends AnyInference,
	OutputSpecs extends Record<string, IBuffable>,
> = Inferences & DeducedInference<OutputSpecs[keyof OutputSpecs]>

type OutputDescription<Inferences extends AnyInference> = {
	entries: OutputEntryDescription[]
	inferences: Inferences
}

export class OutputBindings<
	Inferences extends AnyInference,
	OutputSpecs extends Record<string, IBuffable>,
> extends Bindings<SubInferences<Inferences, OutputSpecs>> {
	public readonly wgslEntries: Record<string, WgslEntry>
	private readonly outputSpecs: { name: string; buffable: IBuffable<Inferences> }[]
	constructor(outputSpecs: OutputSpecs) {
		super()
		// TODO allow input/output?
		this.wgslEntries = wgslEntries(outputSpecs)
		this.outputSpecs = Object.entries(outputSpecs).map(([name, buffable]) => {
			if (!isBuffable(buffable)) throw new ParameterError(`Bad value for output \`${name}\``)
			return {
				name,
				buffable: buffable as IBuffable<Inferences>,
			}
		})
	}
	init() {
		return this.outputSpecs.map(({ name, buffable }) => layoutGroupEntry(name, buffable, false))
	}
	private readonly callInfo = new WeakMap<{}, OutputDescription<Inferences>>()
	entries(inputs: {}, inferences: Inferences) {
		const { device, outputSpecs } = this
		const entries = outputSpecs.map(({ name, buffable }) =>
			outputGroupEntry(
				device,
				name,
				resolvedSize(buffable.sizes, inferences),
				buffable.elementByteSize(inferences)
			)
		)
		this.callInfo.set(inputs, { entries, inferences })
		return entries.map(({ resource }) => ({ resource }))
	}

	encoder(inputs: {}, commandEncoder: GPUCommandEncoder) {
		const { entries } = this.callInfo.get(inputs)!
		for (const { encoder } of entries) encoder(commandEncoder)
	}
	async read(inputs: {}): Promise<Outputs<OutputSpecs>> {
		const { entries, inferences } = this.callInfo.get(inputs)!
		this.callInfo.delete(inputs)
		const buffers = await Promise.all(entries.map(({ read }) => read()))
		return Object.fromEntries(
			this.outputSpecs.map(({ name, buffable }, i) => [
				name,
				buffable.readArrayBuffer(buffers[i], inferences),
			])
		) as Outputs<OutputSpecs>
	}
}

export default function outputs<
	Inferences extends AnyInference,
	OutputSpecs extends Record<string, IBuffable<Inferences>>,
>(outputSpecs: OutputSpecs) {
	return new OutputBindings<Inferences, OutputSpecs>(outputSpecs)
}
