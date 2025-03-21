import type { AnyInference, SizeSpec } from '../inference'

export interface BindingEntryDescription {
	declaration: string
	layoutEntry: GPUUnboundGroupLayoutEntry
}
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
export type GPUUnboundGroupEntry = Omit<GPUBindGroupEntry, 'binding'>
export type GPUUnboundGroupLayoutEntry = Omit<GPUBindGroupLayoutEntry, 'binding'>

export type BoundTypes<BG> = BG extends Bindings<any>
	? {
			inputs: Parameters<BG['entries']>[0]
			outputs: Awaited<ReturnType<BG['read']>>
			inferences: BG['addedInferences']
		}
	: never

export type BindingType<Inferences extends AnyInference> = {
	new (...args: any[]): Bindings<Inferences>
}

export interface WgslEntry<Inferences extends AnyInference = any> {
	sizes: SizeSpec<Inferences>[]
}

export abstract class Bindings<Inferences extends AnyInference> {
	private deviceRef?: WeakRef<GPUDevice>
	public readonly addedInferences: {} = {}
	// TODO: List inferences (provided/needed) in order to order groups ?

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
	setScope(device: GPUDevice, inferences: Inferences, reasons: Record<string, string>) {
		if (this.deviceRef) throw new Error('Binding group already initialized')
		this.deviceRef = new WeakRef(device)
		const entryDescriptors = this.init(inferences, reasons)
		this.staticGenerated = {
			declarations: entryDescriptors.map(({ declaration }) => declaration),
			layoutEntries: entryDescriptors.map(({ layoutEntry }) => layoutEntry),
		}
	}

	// #region To override

	/**
	 * Specifies the names used in the wgsl code
	 */
	public abstract readonly wgslEntries: Record<string, WgslEntry<Inferences>>
	protected init(
		inferences: Inferences,
		reasons: Record<string, string>
	): BindingEntryDescription[] {
		return []
	}
	entries(
		inputs: {},
		inferences: AnyInference,
		reasons: Record<string, string>
	): GPUUnboundGroupEntry[] {
		return []
	}
	encoder(inputs: {}, commandEncoder: GPUCommandEncoder) {}
	read(inputs: {}): {} {
		return {}
	}

	// #endregion
}
