import type { AnyInference } from '../inference'
import { Buffable } from './buffable'
import type { Writer } from './io'

interface StructureElement<Inferences extends AnyInference> {
	name: string
	offset: number
	type: Buffable<Inferences>
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
	get bytesPerAtomic(): number {
		const desc = this.paddedDescriptor
		if (!desc.length) return 0
		const last = desc[desc.length - 1]
		const end = last.offset + last.type.bytesPerAtomic
		return end + this.paddingSize(end)
	}
	writer(buffer: ArrayBuffer): Writer<any> {
		const stride = this.bytesPerAtomic
		const target = new Uint8Array(buffer)
		const fields = this.paddedDescriptor.map(({ name, offset, type }) => {
			const tmp = new ArrayBuffer(type.bytesPerAtomic)
			return { name, offset, write: type.writer(tmp), bytes: new Uint8Array(tmp) }
		})
		return (index: number, value: any) => {
			const base = index * stride
			for (const { name, offset, write, bytes } of fields) {
				write(0, value[name])
				target.set(bytes, base + offset)
			}
		}
	}
	reader(buffer: ArrayBuffer): (index: number) => any {
		const stride = this.bytesPerAtomic
		const source = new Uint8Array(buffer)
		const fields = this.paddedDescriptor.map(({ name, offset, type }) => {
			const tmp = new ArrayBuffer(type.bytesPerAtomic)
			return { name, offset, read: type.reader(tmp), bytes: new Uint8Array(tmp) }
		})
		return (index: number) => {
			const base = index * stride
			const result: Record<string, any> = {}
			for (const { name, offset, read, bytes } of fields) {
				bytes.set(source.subarray(base + offset, base + offset + bytes.length))
				result[name] = read(0)
			}
			return result
		}
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
	//@cached()
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
			({ name, type, alignment }) => `@align(${alignment}) ${name}: ${type.wgslSpecification},`
		)
		return `struct ${this.name} {\n\t${members.join('\n\t')}\n}`
	}
}
