import type { BindingType, Bindings } from '../binding'
import type { Buffable } from '../buffers'
import { type AnyInference, extractInference, specifyInferences } from '../inference'
import { workgroupSize } from '../typedArrays'
import { customBindGroupIndex, layoutGroupEntry, outputBindGroupIndex } from './io'

export function kernelScope<Inferences extends AnyInference>(
	compute: string,
	kernelDefaults: Partial<Record<keyof Inferences, number>>,
	{
		device,
		outputs,
		inferences,
		workGroupSize,
		definitions,
		groups,
		bindingsOrder,
	}: {
		device: GPUDevice
		outputs: Record<string, Buffable<Inferences>>
		inferences: Inferences
		workGroupSize: [number, number, number] | null
		definitions: readonly string[]
		groups: Bindings<Inferences>[]
		bindingsOrder: BindingType<Inferences>[]
	}
) {
	// #region Output

	const outputBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
	const outputBindGroupDescription: string[] = []
	const outputDescription: { name: string; binding: number; buffable: Buffable }[] = []

	for (const [name, buffable] of Object.entries(outputs)) {
		const binding = outputBindGroupLayoutEntries.length
		const { layoutEntry, description } = layoutGroupEntry(
			name,
			buffable,
			outputBindGroupIndex,
			binding,
			false
		)
		outputBindGroupLayoutEntries.push(layoutEntry)
		outputBindGroupDescription.push(description)
		outputDescription.push({ name, binding, buffable })
	}
	const outputBindGroupLayout = device.createBindGroupLayout({
		label: 'output-bind-group-layout',
		entries: outputBindGroupLayoutEntries,
	})

	// #endregion Output

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

	const customBindGroupLayout = device.createBindGroupLayout({
		label: 'custom-bind-group-layout',
		entries: orderedGroups
			.flatMap(({ statics: { layoutEntries } }) => layoutEntries)
			.map((layoutEntry, binding) => ({ ...layoutEntry, binding })),
	})
	const customDeclarations = orderedGroups
		.flatMap(({ statics: { declarations } }) => declarations)
		.map((wgsl, binding) => `@group(${customBindGroupIndex}) @binding(${binding}) ${wgsl}`)

	const code = /*wgsl*/ `
${outputBindGroupDescription.join('\n')}
${customDeclarations.join('\n')}

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
			bindGroupLayouts: [outputBindGroupLayout, customBindGroupLayout],
		}),
		compute: { module: shaderModule, entryPoint: 'main' },
	})

	return {
		code,
		kernelInferences,
		shaderModuleCompilationInfo,
		outputBindGroupLayout,
		kernelWorkGroupSize,
		outputDescription,
		pipeline,
		customBindGroupLayout,
		orderedGroups,
	}
}
