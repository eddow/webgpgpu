import type { Buffable } from '../dataTypes'
import { type RequiredAxis, type WorkSizeInfer, applyDefaultInfer } from '../typedArrays'
import type { BoundDataEntry } from '../webgpgpu'
import { workgroupSize } from '../workGroup'
import {
	commonBindGroupIndex,
	inputBindGroupIndex,
	layoutGroupEntry,
	outputBindGroupIndex,
	reservedBindGroupIndex,
} from './io'
export type KernelScope = ReturnType<typeof kernelScope>
export function kernelScope(
	compute: string,
	kernelWorkInf: WorkSizeInfer,
	kernelRequiredInf: RequiredAxis,
	{
		device,
		commonData,
		inputs,
		outputs,
		workSizeInfer,
		workGroupSize,
		definitions,
		reservedBindGroupLayout,
	}: {
		device: GPUDevice
		commonData: readonly BoundDataEntry[]
		inputs: Record<string, Buffable>
		outputs: Record<string, Buffable>
		workSizeInfer: WorkSizeInfer
		workGroupSize: [number, number, number] | null
		definitions: readonly string[]
		reservedBindGroupLayout: GPUBindGroupLayout
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
	const inputsDescription: [string, number, Buffable][] = []

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

	const kernelWorkSizeInfer = applyDefaultInfer(
		workSizeInfer,
		kernelWorkInf,
		kernelRequiredInf,
		'Kernel definition'
	)
	const kernelWorkGroupSize =
		workGroupSize ||
		workgroupSize([kernelWorkSizeInfer.x, kernelWorkSizeInfer.y, kernelWorkSizeInfer.z], device)

	const code = /*wgsl*/ `
@group(${reservedBindGroupIndex}) @binding(0) var<uniform> threads : vec3u;
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
				reservedBindGroupLayout,
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
		kernelWorkSizeInfer,
		shaderModuleCompilationInfo,
		inputBindGroupLayout,
		outputBindGroupLayout,
		kernelWorkGroupSize,
		reservedBindGroupLayout,
		outputDescription,
		pipeline,
		commonBindGroup,
	}
}
