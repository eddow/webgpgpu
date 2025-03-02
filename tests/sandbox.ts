type Input<
	Element,
	Depth extends number,
	Accumulator extends unknown[] = [],
> = Accumulator['length'] extends Depth
	? Element
	: ArrayLike<Input<Element, Depth, [unknown, ...Accumulator]>>

// Example usage:
type Test0 = Input<number, 0> // number
type Test1 = Input<number, 1> // ArrayLike<number>
type Test2 = Input<number, 2> // ArrayLike<ArrayLike<number>>
type Test3 = Input<number, 3> // ArrayLike<ArrayLike<ArrayLike<number>>>
type Test4 = Input<number, number> // ArrayLike<any>
