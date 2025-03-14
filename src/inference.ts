import type { IBuffable } from './buffable'
import { InferenceValidationError, ParameterError } from './types'

// #region types

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
export type CreatedInferences<Input> = {} & UnionToIntersection<
	{
		[Name in keyof Input]: ExpandKeys<Name & string, Input[Name]>
	}[keyof Input]
>

export type Inference1 = Inferred
export type Inference2 = readonly [Inferred, Inferred]
export type Inference3 = readonly [Inferred, Inferred, Inferred]
export type Inference4 = readonly [Inferred, Inferred, Inferred, Inferred]
export type Inference = Inference1 | Inference2 | Inference3 | Inference4

export type AnyInference<Key extends string = string> = { [K in Key]: Inferred }

export type StringOnly<ST> = ST extends string ? ST : never
export type InferencesList<SSs extends readonly any[]> = SSs extends readonly [
	infer First,
	...infer Rest, // ✅ Remove extends constraint
]
	? Rest extends readonly SizeSpec<AnyInference>[] // ✅ Apply constraint after inference
		? StringOnly<First> | InferencesList<Rest> // Collect strings
		: StringOnly<First>
	: never
export type DeducedInference<OneBuff extends IBuffable> = Record<
	InferencesList<OneBuff['sizes']>,
	Inferred
>

// #endregion

export function infer<Inferences extends AnyInference, Input extends Record<string, Inference>>(
	inferences: Inferences,
	input: Input,
	reason?: string,
	reasons?: Record<string, string>
): Inferences & CreatedInferences<Input> {
	const setting: AnyInference = {}

	for (const [inference, value] of Object.entries(input))
		if (Array.isArray(value))
			for (let i = 0; i < value.length; i++) setting[`${inference}.${'xyzw'[i]}`] = value[i]
		else setting[inference] = value as Inferred
	return specifyInferences(
		inferences,
		setting as Partial<Inferences>,
		reason,
		reasons
	) as Inferences & CreatedInferences<Input>
}

/**
 * Gives a list of direct values ({'threads.x': 52})
 * @param inferences
 * @param values
 * @param reason
 * @returns
 */
export function specifyInferences<Inferences extends AnyInference>(
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
			if (reason !== undefined && reasons && values[key] !== undefined) reasons[key] = reason
			// @ts-expect-error inferences[...] *can* be undefined
			inferences[key] = values[key]
		}
	return inferences
}

export type SizeSpec<Inferences extends AnyInference = AnyInference> = number | keyof Inferences

export function assertSize<Inferences extends AnyInference>(
	given: number[],
	expected: readonly SizeSpec<Inferences>[],
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

	return specifyInferences(inferences, specified as Partial<Inferences>, reason, reasons)
}

type MapNumbers<TArray extends readonly any[]> = {
	[K in keyof TArray]: number
}

export function resolvedSize<
	Inferences extends AnyInference,
	SizesSpec extends readonly SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
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

export function extractInference<Inferences extends AnyInference>(
	inferences: Inferences,
	name: string,
	dimension: 1,
	defaultInfers?: number
): [number]
export function extractInference<Inferences extends AnyInference>(
	inferences: Inferences,
	name: string,
	dimension: 2,
	defaultInfers?: number
): [number, number]
export function extractInference<Inferences extends AnyInference>(
	inferences: Inferences,
	name: string,
	dimension: 3,
	defaultInfers?: number
): [number, number, number]
export function extractInference<Inferences extends AnyInference>(
	inferences: Inferences,
	name: string,
	dimension: 4,
	defaultInfers?: number
): [number, number, number, number]
export function extractInference<Inferences extends AnyInference>(
	inferences: Inferences,
	name: string,
	dimension: 1 | 2 | 3 | 4,
	defaultInfers?: number
): [number] | [number, number] | [number, number, number] | [number, number, number, number]
export function extractInference<Inferences extends AnyInference>(
	inferences: Inferences,
	name: string,
	dimension: 1 | 2 | 3 | 4,
	defaultInfers?: number
) {
	if (![1, 2, 3, 4].includes(dimension))
		throw new ParameterError(`Invalid inference dimension: ${dimension}`)
	const names =
		dimension === 1
			? [name]
			: 'xyzw'
					.substring(0, dimension)
					.split('')
					.map((c) => `${name}.${c}`)

	return names.map((n) => {
		;(inferences as AnyInference)[n] ??= defaultInfers
		return inferences[n as keyof Inferences]
	})
}

export type Stride<Inferences extends AnyInference> = {
	k: number
	vars: (keyof Inferences)[]
}

export function computeStride<Inferences extends AnyInference>(
	inferences: Inferences,
	sizes: readonly SizeSpec<Inferences>[]
) {
	let stride = { k: 1, vars: [] as (keyof Inferences)[] }
	return [...sizes]
		.reverse()
		.map((ss) => {
			const val = typeof ss === 'string' ? inferences[ss] : (ss as number)
			const rv = stride
			stride =
				val === undefined
					? { k: stride.k, vars: [...stride.vars, ss as keyof Inferences] }
					: { k: stride.k * val, vars: stride.vars }
			return rv
		})
		.reverse() as Stride<Inferences>[]
}
export const wgslStrideCalculus = (stride: Stride<any>) =>
	!stride.vars.length
		? `${stride.k}`
		: stride.vars.join('*') + (stride.k === 1 ? '' : `*${stride.k}`)
