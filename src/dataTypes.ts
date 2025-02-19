export type TypedArray =
	| Float64Array
	| Float32Array
	| BigUint64Array
	| BigInt64Array
	| Uint32Array
	| Int32Array
	| Uint16Array
	| Int16Array
	| Uint8Array
	| Int8Array

abstract class GpGpuData<
	Buffer extends TypedArray,
	FlatArray extends any[],
	Element extends (number | bigint) | (number | bigint)[],
	OriginElement = Element,
> {
	constructor(
		protected bufferType: any,
		protected elementSize: 1 | 2 | 3 | 4
	) {}
	elementConvert?: (element: OriginElement) => Element
	buffered(data: Buffer | OriginElement[] | FlatArray, size: number): Buffer {
		if (data instanceof this.bufferType) return data as Buffer
		if (Array.isArray(data)) {
			const { elementSize, elementConvert } = this
			if (data.length === size && (elementSize !== 1 || elementConvert)) {
				const rv = new this.bufferType(size * elementSize) as Buffer
				let dst = 0
				// Make the `if` early not to make it in the look
				if (elementConvert) {
					if (elementSize === 1)
						for (const element of data as OriginElement[]) {
							rv[dst++] = elementConvert(element) as number | bigint
						}
					else
						for (const element of data as OriginElement[]) {
							rv.set(elementConvert(element) as any[], dst)
							dst += elementSize
						}
				} else
					for (const element of data as Element[]) {
						rv.set(element as any[], dst)
						dst += elementSize
					}
				return rv
			}
			if (data.length === size * elementSize) return new this.bufferType(data)
		}
		throw new Error('Invalid input data')
	}
}
// #region AI-generated

// #region GpGpuX
class GpGpuXFloat64<Element extends (number | bigint) | (number | bigint)[], OriginElement = Element> extends GpGpuData<
	Float64Array,
	number[],
	Element,
	OriginElement
> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(Float64Array, elementSize)
	}
}

class GpGpuXFloat32<Element extends (number | bigint) | (number | bigint)[], OriginElement = Element> extends GpGpuData<
	Float32Array,
	number[],
	Element,
	OriginElement
> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(Float32Array, elementSize)
	}
}
class GpGpuXBigUint64<
	Element extends (number | bigint) | (number | bigint)[],
	OriginElement = Element,
> extends GpGpuData<BigUint64Array, bigint[], Element, OriginElement> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(BigUint64Array, elementSize)
	}
}

class GpGpuXBigInt64<
	Element extends (number | bigint) | (number | bigint)[],
	OriginElement = Element,
> extends GpGpuData<BigInt64Array, bigint[], Element, OriginElement> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(BigInt64Array, elementSize)
	}
}

class GpGpuXUint32<Element extends (number | bigint) | (number | bigint)[], OriginElement = Element> extends GpGpuData<
	Uint32Array,
	number[],
	Element,
	OriginElement
> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(Uint32Array, elementSize)
	}
}

class GpGpuXInt32<Element extends (number | bigint) | (number | bigint)[], OriginElement = Element> extends GpGpuData<
	Int32Array,
	number[],
	Element,
	OriginElement
> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(Int32Array, elementSize)
	}
}

class GpGpuXUint16<Element extends (number | bigint) | (number | bigint)[], OriginElement = Element> extends GpGpuData<
	Uint16Array,
	number[],
	Element,
	OriginElement
> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(Uint16Array, elementSize)
	}
}

class GpGpuXInt16<Element extends (number | bigint) | (number | bigint)[], OriginElement = Element> extends GpGpuData<
	Int16Array,
	number[],
	Element,
	OriginElement
> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(Int16Array, elementSize)
	}
}

class GpGpuXUint8<Element extends (number | bigint) | (number | bigint)[], OriginElement = Element> extends GpGpuData<
	Uint8Array,
	number[],
	Element,
	OriginElement
> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(Uint8Array, elementSize)
	}
}

class GpGpuXInt8<Element extends (number | bigint) | (number | bigint)[], OriginElement = Element> extends GpGpuData<
	Int8Array,
	number[],
	Element,
	OriginElement
> {
	constructor(elementSize: 1 | 2 | 3 | 4) {
		super(Int8Array, elementSize)
	}
}
// #endregion GpGpuX
// #region GpGpu1

export class GpGpuFloat64 extends GpGpuXFloat64<number> {
	constructor() {
		super(1)
	}
}

export class GpGpuFloat32 extends GpGpuXFloat32<number> {
	constructor() {
		super(1)
	}
}

export class GpGpuBigInt64 extends GpGpuXBigInt64<bigint> {
	constructor() {
		super(1)
	}
}

export class GpGpuUint32 extends GpGpuXUint32<number> {
	constructor() {
		super(1)
	}
}

export class GpGpuInt32 extends GpGpuXInt32<number> {
	constructor() {
		super(1)
	}
}

export class GpGpuUint16 extends GpGpuXUint16<number> {
	constructor() {
		super(1)
	}
}

export class GpGpuInt16 extends GpGpuXInt16<number> {
	constructor() {
		super(1)
	}
}

export class GpGpuUint8 extends GpGpuXUint8<number> {
	constructor() {
		super(1)
	}
}

export class GpGpuInt8 extends GpGpuXInt8<number> {
	constructor() {
		super(1)
	}
}

// #endregion GpGpu1
// #region GpGpu2

export class GpGpu2Float64 extends GpGpuXFloat64<[number, number]> {
	constructor() {
		super(2)
	}
}
export class GpGpu2Float32 extends GpGpuXFloat32<[number, number]> {
	constructor() {
		super(2)
	}
}

export class GpGpu2BigUint64 extends GpGpuXBigUint64<[bigint, bigint]> {
	constructor() {
		super(2)
	}
}

export class GpGpu2BigInt64 extends GpGpuXBigInt64<[bigint, bigint]> {
	constructor() {
		super(2)
	}
}

export class GpGpu2Uint32 extends GpGpuXUint32<[number, number]> {
	constructor() {
		super(2)
	}
}

export class GpGpu2Int32 extends GpGpuXInt32<[number, number]> {
	constructor() {
		super(2)
	}
}

export class GpGpu2Uint16 extends GpGpuXUint16<[number, number]> {
	constructor() {
		super(2)
	}
}

export class GpGpu2Int16 extends GpGpuXInt16<[number, number]> {
	constructor() {
		super(2)
	}
}

export class GpGpu2Uint8 extends GpGpuXUint8<[number, number]> {
	constructor() {
		super(2)
	}
}

export class GpGpu2Int8 extends GpGpuXInt8<[number, number]> {
	constructor() {
		super(2)
	}
}

// #endregion GpGpu2
// #region GpGpu3

export class GpGpu3Float64 extends GpGpuXFloat64<[number, number, number]> {
	constructor() {
		super(3)
	}
}

export class GpGpu3Float32 extends GpGpuXFloat32<[number, number, number]> {
	constructor() {
		super(3)
	}
}

export class GpGpu3BigUint64 extends GpGpuXBigUint64<[bigint, bigint, bigint]> {
	constructor() {
		super(3)
	}
}

export class GpGpu3BigInt64 extends GpGpuXBigInt64<[bigint, bigint, bigint]> {
	constructor() {
		super(3)
	}
}

export class GpGpu3Uint32 extends GpGpuXUint32<[number, number, number]> {
	constructor() {
		super(3)
	}
}

export class GpGpu3Int32 extends GpGpuXInt32<[number, number, number]> {
	constructor() {
		super(3)
	}
}

export class GpGpu3Uint16 extends GpGpuXUint16<[number, number, number]> {
	constructor() {
		super(3)
	}
}

export class GpGpu3Int16 extends GpGpuXInt16<[number, number, number]> {
	constructor() {
		super(3)
	}
}

export class GpGpu3Uint8 extends GpGpuXUint8<[number, number, number]> {
	constructor() {
		super(3)
	}
}

export class GpGpu3Int8 extends GpGpuXInt8<[number, number, number]> {
	constructor() {
		super(3)
	}
}

// #endregion GpGpu3
// #region GpGpu4

export class GpGpu4Float64 extends GpGpuXFloat64<[number, number, number, number]> {
	constructor() {
		super(4)
	}
}

export class GpGpu4Float32 extends GpGpuXFloat32<[number, number, number, number]> {
	constructor() {
		super(4)
	}
}

export class GpGpu4BigUint64 extends GpGpuXBigUint64<[bigint, bigint, bigint, bigint]> {
	constructor() {
		super(4)
	}
}

export class GpGpu4BigInt64 extends GpGpuXBigInt64<[bigint, bigint, bigint, bigint]> {
	constructor() {
		super(4)
	}
}

export class GpGpu4Uint32 extends GpGpuXUint32<[number, number, number, number]> {
	constructor() {
		super(4)
	}
}

export class GpGpu4Int32 extends GpGpuXInt32<[number, number, number, number]> {
	constructor() {
		super(4)
	}
}

export class GpGpu4Uint16 extends GpGpuXUint16<[number, number, number, number]> {
	constructor() {
		super(4)
	}
}

export class GpGpu4Int16 extends GpGpuXInt16<[number, number, number, number]> {
	constructor() {
		super(4)
	}
}

export class GpGpu4Uint8 extends GpGpuXUint8<[number, number, number, number]> {
	constructor() {
		super(4)
	}
}

export class GpGpu4Int8 extends GpGpuXInt8<[number, number, number, number]> {
	constructor() {
		super(4)
	}
}

// #endregion GpGpu4

// #endregion AI-generated
