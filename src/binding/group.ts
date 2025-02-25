import type { WebGpGpu } from 'webgpgpu'
import type { AnyInference } from '../inference'

export interface BindingGroupEntry {
	declaration: string
	layoutEntry: Omit<GPUBindGroupLayoutEntry, 'binding'>
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
export type BoundTypes<BG> = BG extends BindingGroup<infer Inputs, infer Outputs, infer Inferences>
	? {
			inputs: Inputs
			outputs: Outputs
			inferences: Inferences
		}
	: never
export type BoundInferences<BG extends BindingGroup> = BG['addedInferences']

export abstract class BindingGroup<
	Inputs extends Record<string, any> = Record<string, any>,
	Outputs extends Record<string, any> = Record<string, any>,
	Inferences extends AnyInference = AnyInference,
> {
	deviceRef?: WeakRef<GPUDevice>
	/**
	 * Specifies the names used in the wgsl code
	 */
	public abstract readonly wgslNames: string[]
	public abstract readonly addedInferences: Inferences

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
	protected abstract init(): BindingGroupEntry[]
	setScope(device: GPUDevice, groupId: number) {
		this.deviceRef = new WeakRef(device)
		const entryDescriptors = this.init()
		this.staticGenerated = {
			declarations: entryDescriptors.map(({ declaration }) => declaration),
			layoutEntries: entryDescriptors.map(({ layoutEntry }) => layoutEntry),
		}
	}
	abstract entries(inferences: Inferences, inputs: Inputs): Omit<GPUBindGroupEntry, 'binding'>[]
}
