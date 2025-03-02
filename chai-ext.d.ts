declare global {
	namespace Chai {
		interface Assertion {
			deepArrayEqual(expected: Array<any>): Assertion
		}
	}
}

export {}
