export class WebGpGpuError extends Error {}

export class InferenceValidationError extends WebGpGpuError {
	name = 'InferenceValidationError'
}

export class ParameterError extends WebGpGpuError {
	name = 'ParameterError'
}

export class CompilationError extends WebGpGpuError {
	name = 'CompilationError'
	constructor(public cause: readonly GPUCompilationMessage[]) {
		super('Compilation error')
	}
}

export function mapEntries<From, To>(
	obj: { [key: string]: From },
	fn: (value: From, key: PropertyKey) => To
): { [key: string]: To } {
	return Object.fromEntries(
		Object.entries(obj).map(([key, value]: [PropertyKey, unknown]) => [key, fn(value as From, key)])
	)
}
