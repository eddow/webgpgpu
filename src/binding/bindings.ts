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
export type GPUUnboundGroupEntry = Omit<GPUBindGroupEntry, 'binding'>
export type GPUUnboundGroupLayoutEntry = Omit<GPUBindGroupLayoutEntry, 'binding'>

export type BoundTypes<BG> = BG extends Bindings<any>
	? {
			inputs: EmptyIfUndefined<Parameters<BG['entries']>[1]>
			outputs: {} //TODO
			inferences: BG['addedInferences']
		}
	: never

export type BindingType<Inferences extends AnyInference> = {
	new (...args: any[]): Bindings<Inferences>
}

export abstract class Bindings<Inferences extends AnyInference> {
	private deviceRef?: WeakRef<GPUDevice>
	/**
	 * Specifies the names used in the wgsl code
	 */
	public abstract readonly wgslNames: string[]
	public readonly addedInferences: {} = {}
	// TODO: usedInferences : inferences type-checking
	//public readonly usedInferences: (keyof Inferences)[] = []

	protected get device() {
		if (!this.deviceRef?.deref()) throw new Error('Binding group disposed / not initialized')
		return this.deviceRef.deref()!
	}
	private staticGenerated?: {
		declarations: string[]
		layoutEntries: GPUUnboundGroupLayoutEntry[]
		definitions?: string
	}
	public get statics() {
		if (!this.staticGenerated) throw new Error('Binding group not initialized')
		return this.staticGenerated
	}
	protected abstract init(
		inferences: Inferences,
		reasons: Record<string, string>
	): BindingEntryDescription[]
	setScope(device: GPUDevice, inferences: Inferences, reasons: Record<string, string>) {
		if (this.deviceRef) throw new Error('Binding group already initialized')
		this.deviceRef = new WeakRef(device)
		const entryDescriptors = this.init(inferences, reasons)
		this.staticGenerated = {
			declarations: entryDescriptors.map(({ declaration }) => declaration),
			layoutEntries: entryDescriptors.map(({ layoutEntry }) => layoutEntry),
		}
	}
	abstract entries(
		inferences: AnyInference,
		inputs: {},
		reasons: Record<string, string>
	): GPUUnboundGroupEntry[]
}
