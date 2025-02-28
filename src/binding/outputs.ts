import type { OutputType } from 'src/webgpgpu'
import { type AnyInference, type DeducedInference, resolvedSize } from '../inference'
import { ParameterError } from '../types'
import type { Buffable } from '../types/buffable'
import { isBuffable } from '../types/ggData'
import { Bindings } from './bindings'
import { type OutputEntryDescription, layoutGroupEntry, outputGroupEntry } from './io'

type Outputs<Specs extends Record<string, Buffable>> = { [K in keyof Specs]: OutputType<Specs[K]> }

type OutputDescription<Inferences extends AnyInference> = {
	entries: OutputEntryDescription[]
	inferences: Inferences
}

export class OutputBindings<
	Inferences extends AnyInference,
	OutputSpecs extends Record<string, Buffable<Inferences>>,
> extends Bindings<Inferences & DeducedInference<OutputSpecs[keyof OutputSpecs]>> {
	public readonly wgslNames: string[]
	private readonly outputSpecs: { name: string; buffable: Buffable<Inferences> }[]
	constructor(inputSpecs: OutputSpecs) {
		super()
		// TODO allow input/output?
		this.wgslNames = Object.keys(inputSpecs)
		this.outputSpecs = Object.entries(inputSpecs).map(([name, buffable]) => {
			if (!isBuffable(buffable)) throw new ParameterError(`Bad value for output \`${name}\``)
			return {
				name,
				buffable: buffable as Buffable<Inferences>,
			}
		})
	}
	init() {
		return this.outputSpecs.map(({ name, buffable }) => layoutGroupEntry(name, buffable, false))
	}
	private readonly callInfo = new WeakMap<{}, OutputDescription<Inferences>>()
	entries(inferences: Inferences, inputs: {}) {
		const { device, outputSpecs } = this
		const entries = outputSpecs.map(({ name, buffable }) =>
			outputGroupEntry(
				device,
				name,
				resolvedSize(buffable.size, inferences),
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
	OutputSpecs extends Record<string, Buffable<Inferences>>,
>(outputSpecs: OutputSpecs) {
	return new OutputBindings<Inferences, OutputSpecs>(outputSpecs)
}
