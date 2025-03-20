import { cached } from '../hacks'
import type { AnyInference, Inference } from '../inference'
import { Buffable, type IBuffable } from './buffable'
import type { Writer } from './io'

interface StructureElement<Inferences extends AnyInference> {
	name: string
	offset: number
	type: IBuffable<Inferences>
	alignment?: number
}
export class Struct<
	Inferences extends AnyInference,
	// TODO It's possible to have one array as the last member if variable-sized, or many fixed-size arrays.
	Value extends Record<string, Buffable<Inferences, any, []>>,
> extends Buffable<Inferences, any, [], []> {
	get elementSizes() {
		return [] as [] // :-D
	}
	writer(buffer: ArrayBuffer): Writer<any> {
		throw new Error('Method not implemented: Struct.write')
	}
	reader(buffer: ArrayBuffer): (index: number) => any {
		throw new Error('Method not implemented: Struct.read')
	}
	get bytesPerAtomic(): number {
		throw new Error('Method not implemented: Struct.bytesPerAtomic')
	}
	get wgslSpecification() {
		return this.name
	}
	public constructor(
		public readonly name: string,
		public readonly descriptor: Value
	) {
		super([])
	}
	// TODO try packing f32 on align4 - difference of padding between UBO/SBO is *only* in arrays (array/ubo -> pad16/elm)
	private paddingSize(orgSize: number) {
		const frac = orgSize & 15
		return frac ? 16 - frac : 0
	}
	@cached()
	get paddedDescriptor(): StructureElement<Inferences>[] {
		let offset = 0
		return Object.entries(this.descriptor)
			.sort(([_a, a], [_b, b]) => b.bytesPerAtomic - a.bytesPerAtomic)
			.map(([name, type]) => {
				const padding = this.paddingSize(type.bytesPerAtomic)
				const rv = {
					name,
					offset,
					type,
					alignment: 16,
				}
				offset += padding + type.bytesPerAtomic
				return rv
			})
	}
	get base() {
		return this
	}
	with<NewValue extends Record<string, Buffable<Inferences, any, any, []>>>(
		newValue: NewValue,
		newName?: string
	) {
		return new Struct(newName ?? this.name, { ...this.descriptor, ...newValue })
	}
	get wgsl() {
		const members = this.paddedDescriptor.map(
			({ name, type, alignment, offset }) =>
				`@align(${alignment}) @offset(${offset}) var ${name}: ${type.wgslSpecification};`
		)
		return `struct ${this.name} {
	${members.join('\n\t')}
};`
	}
}
