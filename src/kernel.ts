import type { BindingType, Bindings, WgslEntry } from './binding'
import type { BufferReader } from './buffable'
import { elements } from './hacks'
import {
	type AnyInference,
	computeStride,
	extractInference,
	specifyInferences,
	wgslStrideCalculus,
} from './inference'
import { log } from './log'
import { type AnyInput, CompilationError } from './types'
import { workGroupCount, workgroupSize } from './workgroup'

export function makeKernel<
	Inferences extends AnyInference,
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
>(
	compute: string,
	constants: Record<string, GPUPipelineConstantValue> | undefined,
	device: GPUDevice,
	inferences: Inferences,
	workGroupSize: [number, number, number] | null,
	declarations: string[],
	computations: string[],
	initializations: string[],
	groups: Bindings<Inferences>[],
	bindingsOrder: BindingType<Inferences>[],
	wgslNames: Record<string, WgslEntry<Inferences>>
) {
	const kernelInferences = { ...inferences }
	// TODO: ensure threads.# who have not been used are fixed to 1 (1- find a way to note which inference has been used)
	// exp. to-do: If no inference on `threads.y` for instance, don't "parallelize" on thread.y, it will be defaulted to 1
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
	const bindingsDeclarations = orderedGroups
		.flatMap(({ statics: { declarations } }) => declarations)
		.map((wgsl, binding) => `@group(${0}) @binding(${binding}) ${wgsl}`)

	const strides = Object.entries(wgslNames)
		.filter(([_, { sizes }]) => sizes.length >= 2)
		.map(([name, { sizes }]) => {
			const strides = computeStride(kernelInferences, sizes)
			return (
				strides.every((stride) => stride.vars.length === 0)
					? {
							declaration: `const ${name}Stride = vec${sizes.length}u(${elements(strides, 'k').join(', ')});`,
						}
					: {
							declaration: `var<private> ${name}Stride: vec${sizes.length}u;`,
							calculus: `${name}Stride = vec${sizes.length}u(${strides.map(wgslStrideCalculus).join(', ')});`,
						}
			) as {
				declaration: string
				calculus?: string
			}
		})

	const code = `
${bindingsDeclarations.join('\n')}
${declarations.join('\n')}
${elements(strides, 'declaration').join('\n')}

@compute @workgroup_size(${kernelWorkGroupSize.join(', ') || '1'})
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
		${elements(strides, 'calculus').join('\n\t\t')}
		${initializations.join('\n\t\t')}
		${compute}
		${computations.join('\n\t\t')}
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
		compute: { module: shaderModule, entryPoint: 'main', constants },
	})
	async function kernel(
		device: GPUDevice,
		inputs: Inputs,
		infers: Partial<Record<keyof Inferences, number>>,
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
		const callReasons = { ...inferenceReasons }
		const callInfer = specifyInferences(
			{ ...kernelInferences },
			infers as Partial<Inferences>,
			'Kernel explicit inference',
			callReasons
		) as Inferences
		// First a flat-map
		const customEntries = orderedGroups.flatMap((group) => {
			const entries = group.entries(inputs, callInfer, callReasons)
			if (entries.length !== group.statics.layoutEntries.length)
				throw Error(
					`BindingGroup entries count (${entries.length}) don't match layout entries length (${group.statics.layoutEntries.length})`
				)
			return entries
		})
		// Expect all callInfer usage are solved
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
		const wgc = workGroupCount(extractInference(callInfer, 'threads', 3, 1), kernelWorkGroupSize)
		passEncoder.dispatchWorkgroups(...wgc)
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
