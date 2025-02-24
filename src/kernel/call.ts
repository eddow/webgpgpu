import type { BufferReader } from '../buffers'
import { type Inferred, defaultedInference, resolvedSize, specifyInference } from '../inference'
import { log } from '../log'
import { workGroupCount } from '../typedArrays'
import { type AnyInput, CompilationError, ParameterError } from '../types'
import {
	type OutputEntryDescription,
	commonBindGroupIndex,
	inputBindGroupIndex,
	inputGroupEntry,
	outputBindGroupIndex,
	outputGroupEntry,
	reservedBindGroupIndex,
} from './io'
import type { KernelScope } from './scope'

export async function callKernel<
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
	Inferences extends Record<string, Inferred>,
>(
	device: GPUDevice,
	inputs: Inputs,
	defaultInfers: Partial<Record<keyof Inferences, number>>,
	{
		inputsDescription,
		kernelInferences,
		shaderModuleCompilationInfo,
		inputBindGroupLayout,
		outputBindGroupLayout,
		kernelWorkGroupSize,
		reservedBindGroupLayout,
		outputDescription,
		pipeline,
		commonBindGroup,
	}: KernelScope
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
		specifyInference(kernelInferences, defaultInfers) as Inferences
	)

	// #region Input

	const usedInputs = new Set<string>()
	const inputBindGroupEntries: GPUBindGroupEntry[] = []
	for (const [name, binding, buffable] of inputsDescription) {
		usedInputs.add(name)
		// TODO: default values
		if (!inputs[name]) throw new ParameterError(`Missing input: ${name}`)
		const typeArray = buffable.toTypedArray<Inferences>(
			callInfer,
			inputs[name]!,
			`input \`${name}\``,
			{}
		)
		const resource = inputGroupEntry(
			device,
			name,
			resolvedSize(buffable.size, callInfer),
			typeArray
		)
		inputBindGroupEntries.push({
			binding,
			resource,
		})
	}
	const unusedInputs = Object.keys(inputs).filter((name) => !usedInputs.has(name))
	if (unusedInputs.length) log.warn(`Unused inputs: ${unusedInputs.join(', ')}`)
	const inputBindGroup = device.createBindGroup({
		label: 'input-bind-group',
		layout: inputBindGroupLayout,
		entries: inputBindGroupEntries,
	})

	// #endregion
	// #region Reserved bind group

	const explicit = [
		callInfer['threads.x'] ?? 1,
		callInfer['threads.y'] ?? 1,
		callInfer['threads.z'] ?? 1,
	] as [number, number, number]

	const workGroups = workGroupCount(explicit, kernelWorkGroupSize) as [number, number, number]

	const workSizeBuffer = device.createBuffer({
		size: 16,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	})
	device.queue.writeBuffer(workSizeBuffer, 0, new Uint32Array(explicit))
	const reservedBindGroup = device.createBindGroup({
		label: 'reserved-bind-group',
		layout: reservedBindGroupLayout!,
		entries: [
			{
				binding: 0,
				resource: { buffer: workSizeBuffer },
			},
		],
	})

	// #endregion

	// #region Output

	const outputBindGroupEntries: GPUBindGroupEntry[] = []
	const outputBuffers: OutputEntryDescription[] = []
	for (const { name, binding, buffable } of outputDescription) {
		const OutputEntryDescription = outputGroupEntry(
			device,
			name,
			resolvedSize(buffable.size, callInfer),
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
	passEncoder.setBindGroup(reservedBindGroupIndex, reservedBindGroup)
	passEncoder.setBindGroup(commonBindGroupIndex, commonBindGroup)
	passEncoder.setBindGroup(inputBindGroupIndex, inputBindGroup)
	passEncoder.setBindGroup(outputBindGroupIndex, outputBindGroup)
	passEncoder.dispatchWorkgroups(...workGroups)
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
