import type { AnyInference } from '../inference'
import type { BindingEntryDescription } from './io'

export class FieldsDescriptor<FieldInfo extends {}> {
	public readonly entries: (FieldInfo & { name: string })[] = []
	private readonly indexes: Record<string, number> = {}
	index(name: string) {
		if (!this.indexes[name]) throw new Error(`Unknown name: ${name}`)
		return this.indexes[name]
	}
	add(info: FieldInfo & { name: string }) {
		const { name } = info
		if (this.indexes[name]) throw new Error(`Name conflict: ${name}`)
		this.indexes[name] = this.entries.length
		this.entries.push(info)
	}
}

type EmptyIfUndefined<T> = T extends undefined ? {} : T

export type BoundTypes<BG> = BG extends Bindings
	? {
			inputs: EmptyIfUndefined<Parameters<BG['entries']>[1]>
			outputs: {} //TODO
			inferences: BG['addedInferences']
		}
	: never

export abstract class Bindings {
	deviceRef?: WeakRef<GPUDevice>
	/**
	 * Specifies the names used in the wgsl code
	 */
	public abstract readonly wgslNames: string[]
	public readonly addedInferences: {} = {}

	protected get device() {
		if (!this.deviceRef?.deref()) throw new Error('Binding group disposed / not initialized')
		return this.deviceRef.deref()!
	}
	private staticGenerated?: {
		declarations: string[]
		layoutEntries: Omit<GPUBindGroupLayoutEntry, 'binding'>[]
		definitions?: string
	}
	public get statics() {
		if (!this.staticGenerated) throw new Error('Binding group not initialized')
		return this.staticGenerated
	}
	protected abstract init(): BindingEntryDescription[]
	setScope(device: GPUDevice) {
		if (this.deviceRef) throw new Error('Binding group already initialized')
		this.deviceRef = new WeakRef(device)
		const entryDescriptors = this.init()
		this.staticGenerated = {
			declarations: entryDescriptors.map(({ declaration }) => declaration),
			layoutEntries: entryDescriptors.map(({ layoutEntry }) => layoutEntry),
		}
	}
	abstract entries(inferences: AnyInference, inputs: {}): Omit<GPUBindGroupEntry, 'binding'>[]
}
