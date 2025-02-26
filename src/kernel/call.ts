import type { BufferReader } from '../buffers'
import { type AnyInference, extractInference, specifyInferences } from '../inference'
import { log } from '../log'
import { workGroupCount } from '../typedArrays'
import { type AnyInput, CompilationError } from '../types'
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
		kernelWorkGroupSize,
		pipeline,
		customBindGroupLayout,
		orderedGroups,
	}: ReturnType<typeof kernelScope<Inferences>>
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
		layout: customBindGroupLayout,
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
