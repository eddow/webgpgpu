/*
type Inferred = number | undefined
type AnyInference = { [K: string]: Inferred }
type SizeSpec<Inferences extends AnyInference = AnyInference> = number | keyof Inferences

const sizesSpec = [1, 'blah', 2] as const
type SS = typeof sizesSpec
type StringOnly<ST> = ST extends string ? ST : never
type InferencesList<SSs extends readonly SizeSpec<AnyInference>[]> = SSs extends readonly [
	infer First,
	...infer Rest, // ✅ Remove extends constraint
]
	? Rest extends readonly SizeSpec<AnyInference>[] // ✅ Apply constraint after inference
		? StringOnly<First> | InferencesList<Rest> // Collect strings
		: StringOnly<First>
	: never

type IL = InferencesList<SS> // ✅ Correctly resolves to "blah"
*/
function myFunction<const T extends (string | number)[]>(...args: T) {
	return args
}

// Infer a precise type
type InferredType<T extends (string | number)[]> = T[number]

// Example usage
const result = myFunction(1, 'stringA', 'stringB', 42)

// Inferred type: "stringA" | "stringB" | number
type Test = InferredType<typeof result>
