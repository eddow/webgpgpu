declare global {
	namespace Chai {
		interface Assertion {
			typedArrayEqual(expected: number | number[] | number[][] | number[][][]): Assertion
		}
	}
}

export {}
