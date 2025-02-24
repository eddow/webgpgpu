import { Float16Array } from '@petamoriken/float16'
import type { SizeSpec } from 'inference'
import {
	InferenceValidationError,
	type TypedArray,
	type TypedArray0D,
	type TypedArray1D,
	type TypedArray2D,
	type TypedArray3D,
	type TypedArray4D,
	type TypedArrayXD,
	type WorkSize,
} from './types'

export type NumericSizesSpec<SizesSpec extends SizeSpec[]> = {
	[K in keyof SizesSpec]: number
}

export function dimensionedArray(typedArray: TypedArray): TypedArray0D
export function dimensionedArray(typedArray: TypedArray, size: [number]): TypedArray1D
export function dimensionedArray(typedArray: TypedArray, size: [number, number]): TypedArray2D
export function dimensionedArray(
	typedArray: TypedArray,
	size: [number, number, number]
): TypedArray3D
export function dimensionedArray(
	typedArray: TypedArray,
	size: [number, number, number, number]
): TypedArray4D
export function dimensionedArray(
	typedArray: TypedArray,
	size:
		| []
		| [number]
		| [number, number]
		| [number, number, number]
		| [number, number, number, number] = []
	// TODO: elementSize in order to assert?
): TypedArrayXD {
	const elementSize = typedArray.length / size.reduce((a, b) => a * b, 1)
	if (![1, 2, 3, 4, 6, 8, 9, 12, 16].includes(elementSize))
		throw new InferenceValidationError(
			`Array size mismatch: With a length of ${typedArray.length} a size-${size.join('x')} array/texture would have an element size of ${elementSize}`
		)
	return Object.assign(typedArray, {
		elementSize,
		size,
	})
}

export function isTypedArray(value: any): value is TypedArray {
	return (
		value instanceof Float32Array ||
		value instanceof Float16Array ||
		value instanceof Uint32Array ||
		value instanceof Int32Array
	)
}
export function isTypedArrayXD(
	value: any
): value is TypedArray0D | TypedArray1D | TypedArray2D | TypedArray3D | TypedArray4D {
	return (
		isTypedArray(value) && 'size' in value && Array.isArray(value.size) && value.size.length <= 4
	)
}
export function isTypedArray2D(value: any): value is TypedArray2D {
	return (
		isTypedArray(value) && 'size' in value && Array.isArray(value.size) && value.size.length === 2
	)
}
export function isTypedArray3D(value: any): value is TypedArray3D {
	return (
		isTypedArray(value) && 'size' in value && Array.isArray(value.size) && value.size.length === 3
	)
}
export function isTypedArray4D(value: any): value is TypedArray3D {
	return (
		isTypedArray(value) && 'size' in value && Array.isArray(value.size) && value.size.length === 4
	)
}

// TODO: Structs

function ceilDiv2(value: number): number {
	return (value >> 1) + (value & 1)
}
/**
 * Find the best workgroup size for the given working size
 * "best" here means the most divided (biggest workgroup-size) while having the lowest overhead (ceiling when dividing by 2)
 * @param size
 * @param maxSize Maximum parallelization size for workgroups
 * @returns workGroupSize The optimized workgroup size and the corresponding count of workgroups
 */
export function workgroupSize(size: (number | undefined)[], device: GPUDevice): WorkSize {
	const {
		limits: {
			maxComputeInvocationsPerWorkgroup,
			maxComputeWorkgroupSizeX,
			maxComputeWorkgroupSizeY,
			maxComputeWorkgroupSizeZ,
		},
	} = device
	let remainingSize = maxComputeInvocationsPerWorkgroup
	const remainingWorkGroupSize = [
		maxComputeWorkgroupSizeX,
		maxComputeWorkgroupSizeY,
		maxComputeWorkgroupSizeZ,
	]
	const workGroupCount = size.map((v) => v ?? 1) as WorkSize
	const workGroupSize = workGroupCount.map(() => 1) as WorkSize
	while (remainingSize > 1) {
		// Try to find an even dimension (no overhead)
		let chosenIndex = workGroupCount.findIndex(
			(v, i) => v % 2 === 0 && remainingWorkGroupSize[i] > 1
		)
		if (chosenIndex === -1) {
			/*
[5, 7] = 35 : try to parallelize while keeping the overall size as minimal as possible
ceil(size/2)*2 =>
[ceil(5/2), 7] *2 = [3, 7] *2 -> 21*2 = 42
[5, ceil(7/2)] *2 = [5, 4] *2 -> 20*2 = 40 <- this is the best

So, optimal = divide the max(size) if ceiling is involved
*/
			const maxSizeValue = Math.max(
				...workGroupCount.map((v, i) => (remainingWorkGroupSize[i] === 1 ? 0 : v))
			)
			if (maxSizeValue <= 1) break
			chosenIndex = workGroupCount.findIndex(
				(v, i) => v === maxSizeValue && remainingWorkGroupSize[i] > 1
			)
		}
		workGroupCount[chosenIndex] = ceilDiv2(workGroupCount[chosenIndex])
		workGroupSize[chosenIndex] <<= 1
		remainingWorkGroupSize[chosenIndex] >>= 1
		remainingSize >>= 1
	}
	while (remainingSize > 1) {
		// If we remain with available division BUT no more dimension to divide,
		// then divide the not-yet inferred dimension (undefined)
		const notInferredIndex = size.findIndex(
			(v, i) => v === undefined && remainingWorkGroupSize[i] > 1
		)
		if (notInferredIndex === -1) break
		// Here, no ceiling: we play with powers of 2
		const affected = Math.min(
			remainingSize,
			remainingWorkGroupSize[notInferredIndex] / workGroupSize[notInferredIndex]
		)
		remainingSize /= affected
		workGroupSize[notInferredIndex] *= affected
		remainingWorkGroupSize[notInferredIndex] /= affected
	}
	return workGroupSize
}

export function workGroupCount(workSize: WorkSize, workGroupSize: WorkSize): WorkSize {
	return workSize.map((v, i) => Math.ceil(v / (workGroupSize[i] ?? 1))) as WorkSize
}

export function explicitWorkSize(workSize: [] | WorkSize) {
	return Array.from({ length: 3 }, (_, i) => workSize[i] ?? 1) as [number, number, number]
}
