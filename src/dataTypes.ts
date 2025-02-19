import type { Float16Array } from '@petamoriken/float16'

// With manually removed types not supported by WebGPU
//| Float64Array
//| BigUint64Array
//| BigInt64Array
//| Uint16Array
//| Int16Array
//| Uint8Array
//| Int8Array
export type TypedArray = Float32Array | Float16Array | Uint32Array | Int32Array

// TODO: Structs

export abstract class GpGpuData<
	Buffer extends TypedArray,
	FlatArray extends any[],
	Element extends number | number[],
	OriginElement = Element,
> {
	constructor(
		protected readonly bufferType: any,
		public readonly elementSize: number,
		public readonly wgslSpecification: string
	) {}
	protected elementConvert?: (element: OriginElement) => Element
	protected elementRecover?: (element: Element) => OriginElement
	writeArray(data: Buffer | OriginElement[] | FlatArray, size: number): Buffer {
		if (data instanceof this.bufferType) return data as Buffer
		if (Array.isArray(data)) {
			const { elementSize, elementConvert } = this
			if (data.length === size && (elementSize !== 1 || elementConvert)) {
				const rv = new this.bufferType(size * elementSize) as Buffer
				let dst = 0
				// Make the `if` early not to not make it in the loop
				if (elementConvert) {
					if (elementSize === 1)
						for (const element of data as OriginElement[]) {
							rv[dst++] = elementConvert(element) as number
						}
					else
						for (const element of data as OriginElement[]) {
							rv.set(elementConvert(element) as any[], dst)
							dst += elementSize
						}
				} else if (elementSize === 1)
					for (const element of data as OriginElement[]) {
						rv[dst++] = element as number
					}
				else
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
	*readArray(data: Buffer): Generator<OriginElement> {
		const { elementSize, elementRecover } = this
		// Make the `if` early not to not make it in the loop
		if (elementRecover) {
			if (elementSize === 1) {
				for (let i = 0; i < data.length; i++) yield elementRecover(data.at(i) as Element)
			} else {
				const nbrElements = data.length / elementSize
				for (let i = 0; i < nbrElements; i++)
					yield elementRecover(
						Array.from(data.subarray(i * elementSize, (i + 1) * elementSize)) as Element
					)
			}
		} else if (elementSize === 1) {
			for (let i = 0; i < data.length; i++) yield data[i] as OriginElement
		} else {
			const nbrElements = data.length / elementSize
			for (let i = 0; i < nbrElements; i++)
				yield data.subarray(i * elementSize, (i + 1) * elementSize) as OriginElement
		}
	}
	writeUnique(data: Buffer | OriginElement | FlatArray): Buffer {
		return this.writeArray(
			Array.isArray(data) || data instanceof this.bufferType
				? (data as FlatArray)
				: [data as OriginElement],
			1
		)
	}
	readUnique(data: Buffer): OriginElement {
		const { elementSize, elementRecover } = this
		if (elementRecover) return elementRecover((elementSize === 1 ? data.at(0) : data) as Element)
		if (elementSize === 1) return data[0] as OriginElement
		return Array.from(data) as OriginElement
	}
}

export type ArrayValue<GGData> = GGData extends GpGpuData<
	infer Buffer,
	infer FlatArray,
	any,
	infer OriginElement
>
	? Buffer | OriginElement[] | FlatArray
	: never
export type UniqueValue<GGData> = GGData extends GpGpuData<
	infer Buffer,
	infer FlatArray,
	any,
	infer OriginElement
>
	? Buffer | OriginElement | FlatArray
	: never

export class GpGpuXFloat32<
	Element extends number | number[],
	OriginElement = Element,
> extends GpGpuData<Float32Array, number[], Element, OriginElement> {
	constructor(elementSize: number, wgslSpecification: string) {
		super(Float32Array, elementSize, wgslSpecification)
	}
}

export class GpGpuXFloat16<
	Element extends number | number[],
	OriginElement = Element,
> extends GpGpuData<Float16Array, number[], Element, OriginElement> {
	constructor(elementSize: number, wgslSpecification: string) {
		super(Uint16Array, elementSize, wgslSpecification)
	}
}

export class GpGpuXUint32<
	Element extends number | number[],
	OriginElement = Element,
> extends GpGpuData<Uint32Array, number[], Element, OriginElement> {
	constructor(elementSize: number, wgslSpecification: string) {
		super(Uint32Array, elementSize, wgslSpecification)
	}
}

export class GpGpuXInt32<
	Element extends number | number[],
	OriginElement = Element,
> extends GpGpuData<Int32Array, number[], Element, OriginElement> {
	constructor(elementSize: number, wgslSpecification: string) {
		super(Int32Array, elementSize, wgslSpecification)
	}
}
