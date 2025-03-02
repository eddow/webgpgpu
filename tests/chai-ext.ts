import { Float16Array } from '@petamoriken/float16'
import chai from 'chai'
// Recursive function to compare nested structures
function arrayLikeContentEqual(actual: ArrayLike<any>, expected: Array<any>): boolean {
	if (!Array.isArray(expected)) return actual === expected // Direct number comparison

	if (actual.length !== expected.length) return false
	return expected.every((val, i) => arrayLikeContentEqual(actual[i], val))
}

chai.Assertion.addMethod('deepArrayEqual', function (expected: Array<any>) {
	const actual = this._obj

	this.assert(
		arrayLikeContentEqual(actual, expected),
		'expected #{this} to have the same content as #{exp}',
		'expected #{this} to not have the same content as #{exp}',
		expected,
		'length' in actual
			? Array.from(actual) // Convert for better error messages
			: actual
	)
})
