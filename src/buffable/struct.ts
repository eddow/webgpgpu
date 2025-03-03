import type { AnyInference } from '../inference'
import { Buffable } from './buffable'
import type { Writer } from './io'
// TODO: Type inference for Element
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
	get paddedDescriptor() {
		// TODO Struct.paddedDescriptor
		return this.descriptor
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
		return `struct ${this.name} {
	${Object.entries(this.paddedDescriptor)
		.map(([name, value]) => `${name}: ${value.wgslSpecification},`)
		.join('\n')}
};`
	}
}
