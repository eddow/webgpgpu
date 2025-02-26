import type { BindingType, Bindings } from './binding'
import type { BufferReader } from './buffers'
import { type AnyInference, extractInference, specifyInferences } from './inference'
import { log } from './log'
import { workgroupSize } from './typedArrays'
import { workGroupCount } from './typedArrays'
import { type AnyInput, CompilationError } from './types'

export function kernelScope<
	Inferences extends AnyInference,
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
>(
	compute: string,
	kernelDefaults: Partial<Record<keyof Inferences, number>>,
	device: GPUDevice,
	inferences: Inferences,
	workGroupSize: [number, number, number] | null,
	definitions: readonly string[],
	groups: Bindings<Inferences>[],
	bindingsOrder: BindingType<Inferences>[]
) {
	const kernelInferences = specifyInferences(
		{ ...inferences },
		kernelDefaults as Partial<Inferences>
	)
	// Extract `threads` *with* its `undefined` values for design-time workgroupSize optimization
	const kernelWorkGroupSize =
		workGroupSize || workgroupSize(extractInference(kernelInferences, 'threads', 3), device)

	function order(binding: BindingType<Inferences>) {
		const index = bindingsOrder.indexOf(binding)
		return index === -1 ? bindingsOrder.length : index
	}
	const orderedGroups = [...groups].sort(
		(a, b) =>
			order(a.constructor as BindingType<Inferences>) -
			order(b.constructor as BindingType<Inferences>)
	)

	const bindGroupLayout = device.createBindGroupLayout({
		label: 'custom-bind-group-layout',
		entries: orderedGroups
			.flatMap(({ statics: { layoutEntries } }) => layoutEntries)
			.map((layoutEntry, binding) => ({ ...layoutEntry, binding })),
	})
	const declarations = orderedGroups
		.flatMap(({ statics: { declarations } }) => declarations)
		.map((wgsl, binding) => `@group(${0}) @binding(${binding}) ${wgsl}`)

	const code = `
${declarations.join('\n')}

${definitions.join('\n')}

@compute @workgroup_size(${kernelWorkGroupSize.join(',') || '1'})
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
		${compute}
	}
}
`
	//Compile the shader module
	const shaderModule = device.createShaderModule({ code })
	const shaderModuleCompilationInfo = shaderModule.getCompilationInfo()
	//Create pipeline
	const pipeline = device.createComputePipeline({
		label: 'compute-pipeline',
		layout: device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout],
		}),
		compute: { module: shaderModule, entryPoint: 'main' },
	})
	async function kernel(
		device: GPUDevice,
		inputs: Inputs,
		defaultInfers: Partial<Record<keyof Inferences, number>>,
		inferenceReasons: Record<string, string>
	): Promise<Outputs> {
		const messages = (await shaderModuleCompilationInfo).messages
		if (messages.length > 0) {
			let hasError = false

			for (const msg of messages) {
				const formatted = `[${msg.lineNum}:${msg.linePos}] ${msg.message}`
				if (msg.type === 'error') {
					hasError = true
					log.error(formatted)
				} else log.warn(formatted)
			}

			if (hasError) throw new CompilationError(messages)
		}
		// Inference can be done here as non-compulsory inference are not compelling
		const callInfer = specifyInferences(
			kernelInferences,
			defaultInfers as Partial<Inferences>
		) as Inferences
		// First a flat-map
		const customEntries = orderedGroups.flatMap((group) => {
			const entries = group.entries(callInfer, inputs, inferenceReasons)
			if (entries.length !== group.statics.layoutEntries.length)
				throw Error(
					`BindingGroup entries count (${entries.length}) don't match layout entries length (${group.statics.layoutEntries.length})`
				)
			return entries
		})

		const customBindGroup = device.createBindGroup({
			label: 'custom-bind-group',
			layout: bindGroupLayout,
			// Second, we count the binding-id
			entries: customEntries.map((entry, binding) => ({ ...entry, binding })),
		})
		// Encode and dispatch work

		const commandEncoder = device.createCommandEncoder({
			label: 'webGpGpu-encoder',
		})
		const passEncoder = commandEncoder.beginComputePass()
		passEncoder.setPipeline(pipeline)
		passEncoder.setBindGroup(0, customBindGroup)
		passEncoder.dispatchWorkgroups(
			...workGroupCount(extractInference(callInfer, 'threads', 3), kernelWorkGroupSize)
		)
		passEncoder.end()

		// Add the "copy from output-buffer to read-buffer" command
		for (const bindings of orderedGroups) bindings.encoder(inputs, commandEncoder)

		// Submit commands
		device.queue.submit([commandEncoder.finish()])

		const reads = await Promise.all(orderedGroups.map((bindings) => bindings.read(inputs)))

		return reads.reduce(
			(acc, read) => ({
				...acc,
				...read,
			}),
			{}
		) as Outputs
	}
	return {
		kernel,
		code,
		kernelInferences,
	}
}
