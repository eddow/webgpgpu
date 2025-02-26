import type { Bindings } from '../binding'
import type { BufferReader } from '../buffers'
import {
	type AnyInference,
	defaultedInference,
	extractInference,
	resolvedSize,
	specifyInferences,
} from '../inference'
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
	device: GPUDevice,
	inputs: Inputs,
	defaultInfers: Partial<Record<keyof Inferences, number>>,
	groups: Bindings[],
	{
		kernelInferences,
		shaderModuleCompilationInfo,
		outputBindGroupLayout,
		kernelWorkGroupSize,
		outputDescription,
		pipeline,
		commonBindGroup,
		customBindGroupLayout,
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
	const callInfer = defaultedInference(
		specifyInferences(kernelInferences, defaultInfers as Partial<Inferences>) as Inferences
	)
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
	const customEntries = groups
		.toReversed()
		.map((group) => {
			const entries = group.entries(callInfer, inputs)
			if (entries.length !== group.statics.layoutEntries.length)
				throw Error(
					`BindingGroup entries count (${entries.length}) don't match layout entries length (${group.statics.layoutEntries.length})`
				)
			return entries
		})
		.reverse()
		.flat()

	const customBindGroup = device.createBindGroup({
		label: 'custom-bind-group',
		layout: customBindGroupLayout,
		entries: customEntries.map((entry, binding) => ({ ...entry, binding })),
	})
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
