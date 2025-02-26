import type { BindingType, Bindings } from '../binding'
import { type AnyInference, extractInference, specifyInferences } from '../inference'
import { workgroupSize } from '../typedArrays'

export function kernelScope<Inferences extends AnyInference>(
	compute: string,
	kernelDefaults: Partial<Record<keyof Inferences, number>>,
	{
		device,
		inferences,
		workGroupSize,
		definitions,
		groups,
		bindingsOrder,
	}: {
		device: GPUDevice
		inferences: Inferences
		workGroupSize: [number, number, number] | null
		definitions: readonly string[]
		groups: Bindings<Inferences>[]
		bindingsOrder: BindingType<Inferences>[]
	}
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

	const customBindGroupLayout = device.createBindGroupLayout({
		label: 'custom-bind-group-layout',
		entries: orderedGroups
			.flatMap(({ statics: { layoutEntries } }) => layoutEntries)
			.map((layoutEntry, binding) => ({ ...layoutEntry, binding })),
	})
	const customDeclarations = orderedGroups
		.flatMap(({ statics: { declarations } }) => declarations)
		.map((wgsl, binding) => `@group(${0}) @binding(${binding}) ${wgsl}`)

	const code = /*wgsl*/ `
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
			bindGroupLayouts: [customBindGroupLayout],
		}),
		compute: { module: shaderModule, entryPoint: 'main' },
	})

	return {
		code,
		kernelInferences,
		shaderModuleCompilationInfo,
		kernelWorkGroupSize,
		pipeline,
		customBindGroupLayout,
		orderedGroups,
	}
}
