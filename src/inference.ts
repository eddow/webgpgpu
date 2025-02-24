import { InferenceValidationError, ParameterError, mapEntries } from './types'

export type Inferred = number | undefined
// Redo the [required] for debug
type ExpandKeys<Name extends string, Value> = Value extends Inferred
	? { [K in Name]: Inferred }
	: Value extends readonly [any]
		? { [K in `${Name}.x`]: Inferred }
		: Value extends readonly [any, any]
			? { [K in `${Name}.x` | `${Name}.y`]: Inferred }
			: Value extends readonly [any, any, any]
				? { [K in `${Name}.x` | `${Name}.y` | `${Name}.z`]: Inferred }
				: Value extends readonly [any, any, any, any]
					? { [K in `${Name}.x` | `${Name}.y` | `${Name}.z` | `${Name}.w`]: Inferred }
					: never
type UnionToIntersection<U> = (U extends any ? (arg: U) => void : never) extends (
	arg: infer I
) => void
	? I
	: never
export type CreatedInferences<Input> = UnionToIntersection<
	{
		[Name in keyof Input]: ExpandKeys<Name & string, Input[Name]>
	}[keyof Input]
>
export type AnyInference = Record<string, Inferred>
export function infer<
	Inferences extends Record<string, Inferred>,
	Input extends Record<
		string,
		| Inferred
		| readonly [Inferred, Inferred]
		| readonly [Inferred, Inferred, Inferred]
		| readonly [Inferred, Inferred, Inferred, Inferred]
	>, // Allow any tuple length
>(inferences: Inferences, input: Input, reason?: string): Inferences & CreatedInferences<Input> {
	const setting: { [key: string]: Inferred } = {}

	for (const [inference, value] of Object.entries(input))
		if (Array.isArray(value))
			for (let i = 0; i < value.length; i++) setting[`${inference}.${'xyzw'[i]}`] = value[i]
		else setting[inference] = value as Inferred
	return specifyInference(inferences, setting as Partial<Inferences>, reason) as Inferences &
		CreatedInferences<Input>
}

/**
 * Gives a list of direct values ({'threads.x': 52})
 * @param inferences
 * @param values
 * @param reason
 * @returns
 */
export function specifyInference<Inferences extends Record<string, Inferred>>(
	inferences: Inferences,
	values: Partial<Inferences>,
	reason?: string,
	reasons?: Record<string, string>
) {
	for (const key in values)
		if (reason !== undefined && reasons && key in reasons) {
			if (inferences[key] !== values[key])
				throw new InferenceValidationError(
					`${key} is inferred twice:\n` +
						`- ${reasons[key]}: ${inferences[key]}\n` +
						`- ${reason}: ${values[key]}`
				)
		} else {
			if (reason !== undefined && reasons) reasons[key] = reason
			// @ts-expect-error inferences[...] *can* be undefined
			inferences[key] = values[key]
		}
	return inferences
}

export function defaultedInference<Inferences extends Record<string, Inferred>>(
	inferences: Inferences,
	dft = 1
): Inferences & Record<keyof Inferences, number> {
	return mapEntries<Inferred, number, Inferences>(
		inferences,
		(value) => value ?? dft
	) as Inferences & Record<keyof Inferences, number>
}

export type SizeSpec<Inferences extends { [key: string]: Inferred } = { [key: string]: Inferred }> =
	| number
	| keyof Inferences

export function assertSize<Inferences extends { [key: string]: Inferred }>(
	given: number[],
	expected: SizeSpec<Inferences>[],
	inferences: Inferences,
	reason: string,
	reasons: Record<string, string>
) {
	if (given.length !== expected.length)
		throw new ParameterError(
			`Dimension mismatch in size comparison: ${given.length}-D size compared to ${expected.length}-D`
		)
	const specified: Partial<Record<keyof Inferences, number>> = {}
	for (let i = 0; i < given.length; i++) {
		if (typeof expected[i] === 'number') {
			if (expected[i] === given[i]) continue
			throw new InferenceValidationError(
				`Size mismatch in on ${'XYZ'[i]}: ${reason} (${given[i]}) !== hard-coded ${expected[i] as number}`
			)
		}
		specified[expected[i] as keyof Inferences] = given[i] as number
		if (!(expected[i] in inferences))
			throw new ParameterError(`${String(expected[i])} is not an inference`)
	}

	return specifyInference(inferences, specified as Partial<Inferences>, reason, reasons)
}

type MapNumbers<TArray extends any[]> = {
	[K in keyof TArray]: number
}

export function resolvedSize<
	Inferences extends Record<string, Inferred>,
	SizesSpec extends SizeSpec<Inferences>[],
>(size: SizesSpec, inferences: Inferences): MapNumbers<SizesSpec> {
	return size.map((s) => {
		const rv = typeof s === 'number' ? s : (inferences[s] as number)
		if (rv === undefined) throw new InferenceValidationError(`${String(s)} is not inferred`)
		return rv
	}) as MapNumbers<SizesSpec>
}

export const infer1D = undefined
export const infer2D = [undefined, undefined] as const
export const infer3D = [undefined, undefined, undefined] as const
export const infer4D = [undefined, undefined, undefined, undefined] as const
export const basicInference = infer({}, { threads: infer3D })

// Test part
const testInference = { ...basicInference }
const furtherInference = infer(basicInference, { tests: infer2D, t1: infer1D })
type T = typeof furtherInference
const worksOk = furtherInference['threads.y']
//@ts-expect-error
const ShouldNotWork = furtherInference['qwe.u']

const spec = specifyInference(basicInference, { 'threads.z': 3 })
const spec2 = specifyInference(basicInference, { 'threads.z': 4 })
