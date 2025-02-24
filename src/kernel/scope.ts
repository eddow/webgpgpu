import type { TypedArray } from 'types'
import type { Buffable } from '../buffable'
import { type AnyInference, type Inferred, specifyInference } from '../inference'
import { workgroupSize } from '../typedArrays'
import type { BoundDataEntry } from '../webgpgpu'
import {
	commonBindGroupIndex,
	inferredBindGroupIndex,
	inputBindGroupIndex,
	layoutGroupEntry,
	outputBindGroupIndex,
} from './io'
export type KernelScope<Inferences extends Record<string, Inferred>> = ReturnType<
	typeof kernelScope<Inferences>
>
export function kernelScope<Inferences extends Record<string, Inferred>>(
	compute: string,
	kernelDefaults: Partial<Record<keyof Inferences, number>>,
	{
		device,
		commonData,
		inputs,
		outputs,
		inferences,
		inferred,
		workGroupSize,
		definitions,
	}: {
		device: GPUDevice
		commonData: readonly BoundDataEntry[]
		inputs: Record<string, Buffable<Inferences>>
		outputs: Record<string, Buffable>
		inferences: Inferences
		inferred: Record<string, number>
		workGroupSize: [number, number, number] | null
		definitions: readonly string[]
	}
) {
	// #region Common

	const commonBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
	const commonBindGroupEntries: GPUBindGroupEntry[] = []
	const commonBindGroupDescription: string[] = []
	for (const { name, resource, type: buffable } of commonData) {
		const binding = commonBindGroupLayoutEntries.length

		const { layoutEntry, description } = layoutGroupEntry(
			name,
			buffable,
			commonBindGroupIndex,
			binding,
			true
		)
		commonBindGroupLayoutEntries.push(layoutEntry)
		commonBindGroupDescription.push(description)
		commonBindGroupEntries.push({
			binding,
			resource,
		})
	}
	const commonBindGroupLayout = device.createBindGroupLayout({
		label: 'common-bind-group-layout',
		entries: commonBindGroupLayoutEntries,
	})

	// #endregion Common

	// #region Input
	const inputBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
	const inputBindGroupDescription: string[] = []
	const inputsDescription: [string, number, Buffable<Inferences>][] = []

	for (const [name, buffable] of Object.entries(inputs)) {
		const binding = inputBindGroupLayoutEntries.length
		const { layoutEntry, description } = layoutGroupEntry(
			name,
			buffable,
			inputBindGroupIndex,
			binding,
			true
		)
		inputBindGroupLayoutEntries.push(layoutEntry)
		inputBindGroupDescription.push(description)
		inputsDescription.push([name, binding, buffable])
	}
	const inputBindGroupLayout = device.createBindGroupLayout({
		label: 'input-bind-group-layout',
		entries: inputBindGroupLayoutEntries,
	})

	// #endregion Input

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

	const inferredBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = []
	const inferredBindGroupDescription: string[] = []
	const inferredEntries: { name: string; dimension: number }[] = []
	for (const [name, dimension] of Object.entries(inferred)) {
		const binding = inferredBindGroupLayoutEntries.length
		const type = [undefined, 'u32', 'vec2u', 'vec3u', 'vec4u'][dimension]
		if (type === undefined) throw new Error(`Invalid inferred dimension ${dimension}`)
		inferredEntries.push({ name, dimension })
		inferredBindGroupDescription.push(
			`@group(${inferredBindGroupIndex}) @binding(${binding}) var<uniform> ${name} : ${type};`
		)
		inferredBindGroupLayoutEntries.push({
			binding,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: 'uniform' },
		})
	}
	const inferredBindGroupLayout: GPUBindGroupLayout = device.createBindGroupLayout({
		label: 'inferred-bind-group-layout',
		entries: inferredBindGroupLayoutEntries,
	})
	const kernelInferences = specifyInference(
		{ ...inferences },
		kernelDefaults as Partial<Inferences>
	)
	const kernelWorkGroupSize =
		workGroupSize ||
		workgroupSize(
			[kernelInferences['threads.x'], kernelInferences['threads.y'], kernelInferences['threads.z']],
			device
		)

	const code = /*wgsl*/ `
${inferredBindGroupDescription.join('\n')}
${commonBindGroupDescription.join('\n')}
${inputBindGroupDescription.join('\n')}
${outputBindGroupDescription.join('\n')}

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
			bindGroupLayouts: [
				inferredBindGroupLayout,
				commonBindGroupLayout,
				inputBindGroupLayout,
				outputBindGroupLayout,
			],
		}),
		compute: { module: shaderModule, entryPoint: 'main' },
	})

	const commonBindGroup = device.createBindGroup({
		label: 'common-bind-group',
		layout: pipeline.getBindGroupLayout(commonBindGroupIndex),
		entries: commonBindGroupEntries,
	})

	return {
		code,
		inputsDescription,
		kernelInferences,
		shaderModuleCompilationInfo,
		inputBindGroupLayout,
		outputBindGroupLayout,
		kernelWorkGroupSize,
		inferredBindGroupLayout,
		inferredEntries,
		outputDescription,
		pipeline,
		commonBindGroup,
	}
}
