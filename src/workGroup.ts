/**
 * Parallelizing loops, these are the imbricated loops sizes
 * @example [10, 20] => for(int x = 0; x < 10; x++) for(int y = 0; y < 20; y++) {...}
 */
export type WorkSize = [number] | [number, number] | [number, number, number]
function ceilDiv2(value: number): number {
	return (value >> 1) + (value & 1)
}
/**
 * Find the best workgroup size for the given working size
 * "best" here means the most divided (biggest workgroup-size) while having the lowest overhead (ceiling when dividing by 2)
 * @param size
 * @param maxSize Maximum parallelization size for workgroups
 * @returns { workGroupSize, workGroupCount } The optimized workgroup size and the corresponding count of workgroups
 */
export function workgroupSize(
	size: WorkSize,
	adapter: GPUAdapter
): { workGroupSize: WorkSize; workGroupCount: WorkSize } {
	const {
		limits: {
			maxComputeInvocationsPerWorkgroup,
			maxComputeWorkgroupSizeX,
			maxComputeWorkgroupSizeY,
			maxComputeWorkgroupSizeZ,
		},
	} = adapter
	let remainingSize = maxComputeInvocationsPerWorkgroup
	const remainingWorkGroupSize = [
		maxComputeWorkgroupSizeX,
		maxComputeWorkgroupSizeY,
		maxComputeWorkgroupSizeZ,
	]
	const workGroupCount = [...size] as WorkSize
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
				(v) => v === maxSizeValue && remainingWorkGroupSize[v] > 1
			)
		}
		workGroupCount[chosenIndex] = ceilDiv2(workGroupCount[chosenIndex])
		workGroupSize[chosenIndex] <<= 1
		remainingWorkGroupSize[chosenIndex] >>= 1
		remainingSize >>= 1
	}
	return { workGroupSize, workGroupCount }
}

export function workGroupCount(workSize: WorkSize, workGroupSize: WorkSize): WorkSize {
	return workSize.map((v, i) => Math.ceil(v / (workGroupSize[i] ?? 1))) as WorkSize
}

export function explicitWorkSize(workSize: [] | WorkSize) {
	return Array.from({ length: 3 }, (_, i) => workSize[i] ?? 1) as [number, number, number]
}
