import type { AnyInference } from '../inference'
import type { Buffable } from './buffable'

export class Struct<
	Inferences extends AnyInference,
	// TODO It's possible to have one array as the last member
	Value extends Record<string, Buffable<Inferences, any, []>>,
> {
	public constructor(public readonly descriptor: Value) {}
	with<NewValue extends Record<string, Buffable<Inferences, any, any, []>>>(newValue: NewValue) {
		return new Struct({ ...this.descriptor, ...newValue })
	}
	wgsl(name: string) {
		return `struct ${name} {
			${Object.entries(this.descriptor)
				.map(([name, value]) => `${name}: ${value.wgslSpecification},`)
				.join('\n')}
		}`
	}
}
