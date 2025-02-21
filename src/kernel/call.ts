import type { BufferReader } from '../buffers'
import { log } from '../log'
import {
	type AnyInput,
	type RequiredAxis,
	WebGpGpuError,
	type WorkSizeInfer,
	applyDefaultInfer,
	resolvedSize,
} from '../typedArrays'
import { workGroupCount } from '../workGroup'
import {
	type OutputEntryDescription,
	ParameterError,
	commonBindGroupIndex,
	dimensionalInput,
	dimensionalOutput,
	inputBindGroupIndex,
	outputBindGroupIndex,
	reservedBindGroupIndex,
} from './io'
import type { KernelScope } from './scope'

export class CompilationError extends WebGpGpuError {
	name = 'CompilationError'
	constructor(public cause: readonly GPUCompilationMessage[]) {
		super('Compilation error')
	}
}

export async function callKernel<
	Inputs extends Record<string, AnyInput>,
	Outputs extends Record<string, BufferReader>,
>(
	device: GPUDevice,
	inputs: Inputs,
	callWorkInf: WorkSizeInfer,
	callRequiredInf: RequiredAxis,
	{
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
	const callWorkSizeInfer = applyDefaultInfer(
		kernelWorkSizeInfer,
		callWorkInf,
		callRequiredInf,
		'Kernel call'
	)

	// #region Input

	const usedInputs = new Set<string>()
	const inputBindGroupEntries: GPUBindGroupEntry[] = []
	for (const [name, binding, buffable] of inputsDescription) {
		usedInputs.add(name)
		// TODO: default values
		if (!inputs[name]) throw new ParameterError(`Missing input: ${name}`)
		const typeArray = buffable.toTypedArray(callWorkSizeInfer, inputs[name]!, `input \`${name}\``)
		const resource = dimensionalInput(
			device,
			name,
			resolvedSize(buffable.size, callWorkSizeInfer),
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

	for (const c of 'xyz') callWorkSizeInfer[c as 'x' | 'y' | 'z'] ??= 1
	const explicit = [callWorkSizeInfer.x, callWorkSizeInfer.y, callWorkSizeInfer.z] as [
		number,
		number,
		number,
	]

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
		const OutputEntryDescription = dimensionalOutput(
			device,
			name,
			resolvedSize(buffable.size, callWorkSizeInfer),
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
		result[name] = buffable.readTypedArray(reads[i], callWorkSizeInfer)
	}

	return result as Outputs
}
