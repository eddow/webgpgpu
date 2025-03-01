declare global {
	namespace Chai {
		interface Assertion {
			typedArrayEqual(
				expected:
					| number
					| readonly number[]
					| readonly number[][]
					| readonly number[][][]
					| readonly number[][][][]
					| readonly number[][][][][]
					| readonly number[][][][][][]
			): Assertion
		}
	}
}

export {}
