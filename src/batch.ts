import type { Bindings, BoundTypes } from './binding'
import type { IBuffable, IBufferReader, ValuedBuffable } from './buffable'
import type { CodeParts } from './code'
import { mapEntries } from './hacks'
import {
	type AnyInference,
	type CreatedInferences,
	type Inferred,
	type SizeSpec,
	specifyInferences,
} from './inference'
import { log } from './log'
import { type AnyInput, type IWebGpGpu, type InputType, type Kernel, WebGpGpuError } from './types'
import type { RootWebGpGpu, WebGpGpu } from './webgpgpu'

export class BatchError extends WebGpGpuError {
	name = 'BatchError'
}

export type RootGGBatch = GGBatch<
	{
		'threads.x': Inferred
		'threads.y': Inferred
	},
	{},
	{}
>

const roots = new WeakMap<RootWebGpGpu, RootGGBatch>()

export function tick(): Promise<void> {
	return new Promise<void>((res) => setTimeout(res))
}

export type BatchKernel<
	Inferences extends AnyInference,
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, IBufferReader>,
> = Kernel<Inferences, Inputs, Outputs> & { executed: Promise<number> }

export class GGBatch<
	Inferences extends AnyInference,
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, IBufferReader>,
> implements IWebGpGpu<Inferences, Inputs, Outputs>
{
	static createRoot(webgpgpu: RootWebGpGpu): RootGGBatch {
		// TODO: assert webgpgpu has no input/output/inference/...
		let batch = roots.get(webgpgpu)
		if (!batch) {
			batch = new GGBatch<{ 'threads.x': Inferred; 'threads.y': Inferred }, {}, {}>(
				webgpgpu,
				{},
				{}
			)
			roots.set(webgpgpu, batch)
		}
		return batch
	}
	private constructor(
		private webgpgpu: WebGpGpu<any, any, any>,
		private inputs: Record<string, IBuffable<Inferences>>,
		private outputs: Record<string, IBuffable<Inferences>>
	) {}
	private clone(webgpgpu: WebGpGpu<any, any, any>) {
		return new GGBatch<Inferences, Inputs, Outputs>(webgpgpu, this.inputs, this.outputs)
	}
	get inferences(): Inferences {
		const { 'threads.z': _, ...inferences } = this.webgpgpu.inferences
		return inferences as Inferences
	}
	get device(): GPUDevice {
		return this.webgpgpu.device
	}
	get disposed(): boolean {
		return this.webgpgpu.disposed
	}
	get f16(): boolean {
		return true
	}
	dispose(): void {
		this.webgpgpu.dispose()
	}
	define(definitions: Record<string, string>) {
		return this.clone(this.webgpgpu.define(definitions))
	}
	code(...codeParts: (CodeParts | string)[]) {
		return this.clone(this.webgpgpu.code(...codeParts))
	}
	import(...imports: PropertyKey[]) {
		return this.clone(this.webgpgpu.import(...imports))
	}
	common<Specs extends Record<string, ValuedBuffable<Inferences>>>(commons: Specs) {
		return this.clone(this.webgpgpu.common(commons))
	}
	workGroup(...size: [] | [number] | [number, number] | [number, number, number]) {
		return this.clone(this.webgpgpu.workGroup(...size))
	}

	bind<BG extends Bindings<Inferences>>(
		group: BG
	): IWebGpGpu<
		Inferences & BoundTypes<BG>['inferences'],
		Inputs & BoundTypes<BG>['inputs'],
		Outputs & BoundTypes<BG>['outputs']
	> {
		throw new Error('Not implemented: Bindable.batch()')
	}
	infer<
		Input extends Record<
			string,
			| Inferred
			| readonly [Inferred, Inferred]
			| readonly [Inferred, Inferred, Inferred]
			| readonly [Inferred, Inferred, Inferred, Inferred]
		>,
	>(input: Input) {
		return new GGBatch<Inferences & CreatedInferences<Input>, Inputs, Outputs>(
			this.webgpgpu.infer(input),
			this.inputs,
			this.outputs
		)
	}
	specifyInference(values: Partial<Inferences>, reason?: string) {
		return this.clone(this.webgpgpu.specifyInference(values, reason))
	}

	input<
		Specs extends Record<
			string,
			IBuffable<Inferences, any, SizeSpec<Inferences>[], SizeSpec<Inferences>[]>
		>,
	>(inputs: Specs) {
		return new GGBatch<Inferences, Inputs & { [K in keyof Specs]: InputType<Specs[K]> }, Outputs>(
			this.webgpgpu.input(mapEntries(inputs, (input) => input.array('threads.z'))),
			Object.assign(this.inputs, inputs),
			this.outputs
		)
	}
	output<
		Specs extends Record<
			string,
			IBuffable<Inferences, any, SizeSpec<Inferences>[], SizeSpec<Inferences>[]>
		>,
	>(outputs: Specs) {
		return new GGBatch<
			Inferences,
			Inputs,
			Outputs & { [K in keyof Specs]: ReturnType<Specs[K]['readArrayBuffer']> }
		>(
			this.webgpgpu.output(mapEntries(outputs, (output) => output.array('threads.z'))),
			this.inputs,
			Object.assign(this.outputs, outputs)
		)
	}

	/**
	 * Batcher
	 * @param constants Constants to pass to the kernel
	 */
	batch(
		constants?: Record<string, number>
	): (start?: Promise<void>) => BatchKernel<Inferences, Inputs, Outputs>
	/**
	 * Batcher
	 * @param compute WGSL code to compile
	 * @param constants Constants to pass to the kernel
	 */
	batch(
		compute?: string,
		constants?: Record<string, GPUPipelineConstantValue>
	): (start?: Promise<void>) => BatchKernel<Inferences, Inputs, Outputs>

	batch(
		compute?: string | Record<string, GPUPipelineConstantValue>,
		constants?: Record<string, GPUPipelineConstantValue>
	) {
		const kernel = this.webgpgpu.kernel(compute as string, constants)
		return (start: Promise<void> = tick()) => {
			const batchInputs = mapEntries(this.inputs, () => [] as ArrayBuffer[])
			const waiting: ((output: Outputs) => void)[] = []
			const inferred = { ...kernel.inferred } as Inferences
			const reasons = { ...this.webgpgpu.inferenceReasons }
			let consumed = false
			const executed = start.then(async () => {
				consumed = true
				if (waiting.length === 0) {
					log.warn('Empty batch consumed.')
					return
				}
				const outputs = await kernel(batchInputs, inferred)
				for (let i = 0; i < waiting.length; ++i)
					waiting[i](mapEntries(outputs, (output: any[]) => output[i]) as Outputs)
				return waiting.length
			})
			return Object.assign(
				(inputs: Inputs, infers: Partial<Record<keyof Inferences, number>> = {}) => {
					if (consumed) throw new BatchError('Batch consumed.')
					specifyInferences(
						inferred,
						infers as Partial<Inferences>,
						'kernel specification',
						reasons
					)
					for (const i in this.inputs) {
						const ba = this.inputs[i].toArrayBuffer(
							inputs[i],
							inferred,
							`Batch input ${i}`,
							reasons
						)
						batchInputs[i].push(ba)
					}
					return new Promise<Outputs>((resolve) => {
						waiting.push((x) => resolve(x))
					})
				},
				{
					toString() {
						return kernel.toString()
					},
					inferred,
					executed,
				}
			)
		}
	}
}
