import { Float16Array } from '@petamoriken/float16'
import chai from 'chai'

const { expect } = chai

// Function to compare a TypedArray to a regular array
type TypedArray = Uint32Array | Int32Array | Float32Array | Float16Array

// Recursive function to compare nested structures
function typedArrayContentEqual(
	actual:
		| TypedArray
		| TypedArray[]
		| TypedArray[][]
		| number
		| number[]
		| number[][]
		| number[][][],
	expected: number | number[] | number[][] | number[][][]
): boolean {
	if (typeof actual === 'number' && typeof expected === 'number') {
		return actual === expected // Direct number comparison
	}

	if (Array.isArray(actual) && Array.isArray(expected)) {
		if (actual.length !== expected.length) return false
		return actual.every((val, i) => typedArrayContentEqual(val, expected[i])) // Recursively check each element
	}

	if (
		actual instanceof Uint32Array ||
		actual instanceof Int32Array ||
		actual instanceof Float32Array ||
		actual instanceof Float16Array
	) {
		if (!Array.isArray(expected)) return false // Mismatch: TypedArray vs non-array

		if (actual.length !== expected.length) return false // Length mismatch

		return actual.every((val, i) => typedArrayContentEqual(val, expected[i])) // Element-wise comparison
	}

	return false // Default case: not comparable
}

chai.Assertion.addMethod(
	'typedArrayEqual',
	function (expected: number | number[] | number[][] | number[][][]) {
		const actual = this._obj

		this.assert(
			typedArrayContentEqual(actual, expected),
			'expected #{this} to have the same content as #{exp}',
			'expected #{this} to not have the same content as #{exp}',
			expected,
			actual instanceof Uint32Array ||
				actual instanceof Int32Array ||
				actual instanceof Float32Array ||
				actual instanceof Float16Array
				? Array.from(actual) // Convert for better error messages
				: actual
		)
	}
)
/*
// ✅ Example Usage
const floatArray = new Float32Array([1.1, 2.2, 3.3])

expect(floatArray).to.typedArrayEqual([1.1, 2.2, 3.3]) // ✅ Passes
*/
