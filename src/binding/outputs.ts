import type { OutputType } from 'src/webgpgpu'
import { isBuffable } from '../buffable'
import type { Buffable } from '../buffers'
import { type AnyInference, resolvedSize } from '../inference'
import { ParameterError } from '../types'
import { Bindings } from './bindings'
import { type OutputEntryDescription, layoutGroupEntry, outputGroupEntry } from './io'

type Outputs<Specs extends Record<string, Buffable>> = { [K in keyof Specs]: OutputType<Specs[K]> }

export class OutputBindings<
	Inferences extends AnyInference,
	OutputSpecs extends Record<string, Buffable<Inferences>>,
> extends Bindings<Inferences> {
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
	private readonly callInfo = new WeakMap<
		{},
		(OutputEntryDescription & { inferences: Inferences })[]
	>()
	entries(inferences: Inferences, inputs: {}, reasons: Record<string, string>) {
		const { device, outputSpecs } = this
		const entryDescriptors = outputSpecs.map(({ name, buffable }) => ({
			...outputGroupEntry(
				device,
				name,
				resolvedSize(buffable.size, inferences),
				buffable.elementSize,
				buffable.bufferType
			),
			inferences,
		}))
		this.callInfo.set(inputs, entryDescriptors)
		return entryDescriptors.map(({ resource }) => ({ resource }))
	}

	encoder(inputs: {}, commandEncoder: GPUCommandEncoder) {
		const entryDescriptors = this.callInfo.get(inputs)!
		for (const { encoder } of entryDescriptors) encoder(commandEncoder)
	}
	async read(inputs: {}): Promise<Outputs<OutputSpecs>> {
		const entryDescriptors = this.callInfo.get(inputs)!
		const buffers = await Promise.all(entryDescriptors.map(({ read }) => read()))
		return Object.fromEntries(
			this.outputSpecs.map(({ name, buffable }, i) => [
				name,
				buffable.readTypedArray(buffers[i], entryDescriptors[i].inferences),
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
