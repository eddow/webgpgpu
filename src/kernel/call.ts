import type { Bindings, inference } from '../binding'
import type { BufferReader } from '../buffers'
import { type AnyInference, extractInference, resolvedSize, specifyInferences } from '../inference'
import { log } from '../log'
import { workGroupCount } from '../typedArrays'
import { type AnyInput, CompilationError, ParameterError } from '../types'
import {
	type OutputEntryDescription,
	commonBindGroupIndex,
	customBindGroupIndex,
	outputBindGroupIndex,
	outputGroupEntry,
} from './io'
import type { kernelScope } from './scope'

export async function callKernel<
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
	Inferences extends AnyInference,
>(
	{
		device,
		inputs,
		defaultInfers,
		inferenceReasons,
	}: {
		device: GPUDevice
		inputs: Inputs
		defaultInfers: Partial<Record<keyof Inferences, number>>
		inferenceReasons: Record<string, string>
	},
	{
		kernelInferences,
		shaderModuleCompilationInfo,
		outputBindGroupLayout,
		kernelWorkGroupSize,
		outputDescription,
		pipeline,
		commonBindGroup,
		customBindGroupLayout,
		orderedGroups,
	}: ReturnType<typeof kernelScope<Inferences>>
) {
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
		layout: customBindGroupLayout,
		entries: customEntries.map((entry, binding) => ({ ...entry, binding })),
	})
	// #region Output

	const outputBindGroupEntries: GPUBindGroupEntry[] = []
	const outputBuffers: OutputEntryDescription[] = []
	for (const { name, binding, buffable } of outputDescription) {
		const OutputEntryDescription = outputGroupEntry(
			device,
			name,
			resolvedSize(buffable.size, callInfer as any),
			buffable.elementSize,
			buffable.bufferType
		)
		const { resource } = OutputEntryDescription
		outputBindGroupEntries.push({
			binding,
			resource,
		})
		outputBuffers.push(OutputEntryDescription)
	}
	const outputBindGroup = device.createBindGroup({
		label: 'output-bind-group',
		layout: outputBindGroupLayout,
		entries: outputBindGroupEntries,
	})

	// #endregion
	// Encode and dispatch work

	const commandEncoder = device.createCommandEncoder({
		label: 'webGpGpu-encoder',
	})
	const passEncoder = commandEncoder.beginComputePass()
	passEncoder.setPipeline(pipeline)
	passEncoder.setBindGroup(commonBindGroupIndex, commonBindGroup)
	passEncoder.setBindGroup(outputBindGroupIndex, outputBindGroup)
	passEncoder.setBindGroup(customBindGroupIndex, customBindGroup)
	passEncoder.dispatchWorkgroups(
		...workGroupCount(extractInference(callInfer, 'threads', 3), kernelWorkGroupSize)
	)
	passEncoder.end()

	// Add the "copy from output-buffer to read-buffer" command
	for (const { encoder } of outputBuffers) encoder(commandEncoder)

	// Submit commands
	device.queue.submit([commandEncoder.finish()])

	const reads = await Promise.all(outputBuffers.map(({ read }) => read()))

	const result: Record<string, BufferReader> = {}
	for (let i = 0; i < outputDescription.length; i++) {
		const { name, buffable } = outputDescription[i]
		result[name] = buffable.readTypedArray(reads[i], callInfer)
	}

	return result as Outputs
}
