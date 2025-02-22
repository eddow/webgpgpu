import type { Buffable } from 'dataTypes'
import { log } from './log'
import {
	type NumericSizesSpec,
	type SizeSpec,
	type WorkSizeInfer,
	assertElementSize,
	assertSize,
	isTypedArrayXD,
	resolvedSize,
} from './typedArrays'
import { ArraySizeValidationError, type TypedArray } from './types'

// Poorly typed but for internal use only
export function elementsToTypedArray<
	Buffer extends TypedArray,
	OriginElement,
	InputSizesSpec extends SizeSpec[],
>(
	specification: Buffable<Buffer, OriginElement, SizeSpec[], InputSizesSpec>,
	workSizeInfer: WorkSizeInfer,
	data: any,
	size: SizeSpec[],
	required?: string
): Buffer {
	const { bufferType, elementSize, elementConvert, transformSize } = specification
	if (data instanceof bufferType && 'elementSize' in data && data.elementSize !== elementSize)
		assertElementSize(data.elementSize, elementSize)
	// #region 0D
	if (size.length === 0) {
		if (data instanceof bufferType) {
			assertElementSize((data as Buffer).length, elementSize)
			return data as Buffer
		}
		const elm = elementConvert
			? elementConvert(data as OriginElement, resolvedSize(transformSize, workSizeInfer))
			: data
		return elm instanceof bufferType ? (elm as Buffer) : new bufferType(elm)
	}
	// #endregion 0D
	// #region 1D
	if (size.length === 1) {
		if (data instanceof bufferType) {
			if ((data as Buffer).length % elementSize !== 0)
				throw new ArraySizeValidationError(
					`Size mismatch in dimension 1: ${data.length} is not a multiple of ${elementSize}`
				)
			assertSize([(data as Buffer).length / elementSize], [size[0]], workSizeInfer, required)
			return data as Buffer
		}
		if (!Array.isArray(data))
			throw new ArraySizeValidationError('Input is not an array nor a typed array')
		assertSize([data.length], [size[0]], workSizeInfer, required)
		const rv = new bufferType(data.length * elementSize) as Buffer
		let dst = 0
		// Make the `if` early not to not make it in the loop
		if (elementConvert) {
			for (const element of data as OriginElement[]) {
				rv.set(
					elementConvert(element, [
						/*todo*/
					] as NumericSizesSpec<InputSizesSpec>),
					dst
				)
				dst += elementSize
			}
		} else
			for (const element of data as number[][]) {
				rv.set(element, dst)
				dst += elementSize
			}
		return rv
	}
	// #endregion 1D
	// #region 2~3-D

	if (data instanceof bufferType) {
		// TODO: multidimensional inferring
		if (!isTypedArrayXD(data))
			throw new ArraySizeValidationError(
				`When giving a ${size.length}-D typed array as input, the input must have given dimension. Use \`dimensionedArray\``
			)
		if (data.size.length !== size.length)
			throw new ArraySizeValidationError(
				`Dimensions mismatch: Typed array of dimension ${data.size.length} is used in a ${size.length}D context`
			)
		assertSize(data.size, size, workSizeInfer, required)
		return data as Buffer
	}
	if (!Array.isArray(data))
		throw new ArraySizeValidationError('Input is not an array nor a typed array')
	assertSize([data.length], size.slice(0, 1), workSizeInfer, required)
	let rv: Buffer | undefined
	let itemSize: number | undefined
	let dst = 0
	for (const element of data) {
		const subBuffer = elementsToTypedArray(
			specification,
			workSizeInfer,
			element,
			size.slice(1),
			required && `[Slice of] ${required}`
		)
		if (!rv) {
			itemSize = subBuffer.length
			// TODO: multidimensional inferring
			rv = new bufferType(resolvedSize(size, workSizeInfer).reduce((a, b) => a * b, 1))
		} else if (subBuffer.length !== itemSize)
			throw new ArraySizeValidationError(
				`Size mismatch in dimension ${size.length}: Buffer length ${subBuffer.length} was expected to be ${itemSize}`
			)
		rv.set(subBuffer, dst)
		dst += itemSize
	}
	if (!rv) log.warn('Size 0 buffer created')
	return rv ?? new bufferType(0)
}

function bufferPosition(index: number[], size: number[], elementSize: number): number {
	let pos = 0
	for (let i = 0; i < size.length; i++) {
		pos *= size[i]
		pos += index[i]
	}
	return pos * elementSize
}

function nextXdIndex(index: number[], size: number[]): boolean {
	for (let i = index.length - 1; i >= 0; i--) {
		index[i]++
		if (index[i] < size[i]) return true
		index[i] = 0
	}
	return false
}

export class BufferReader<
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	SizesSpec extends SizeSpec[] = SizeSpec[],
	InputSizesSpec extends SizeSpec[] = [],
	InputSpec extends number[] = number[],
> {
	constructor(
		public readonly buffable: Buffable<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>,
		public readonly buffer: Buffer,
		public readonly size: number[]
	) {}
	at(...index: InputSpec): OriginElement {
		const {
			size,
			buffer,
			buffable: { elementSize, elementRecover, transformSize },
		} = this
		if (index.length !== size.length)
			throw new ArraySizeValidationError(
				`Index length mismatch: taking ${index.length}-D index element out of ${size.length}-D buffer`
			)
		const pos = bufferPosition(index, size, elementSize)
		const element = buffer.subarray(pos, pos + elementSize)
		// @ts-expect-error TODO: fix size
		return (elementRecover?.(element, [
			/*todo*/
		]) ?? element) as OriginElement
	}
	*entries(): Generator<[InputSpec, OriginElement]> {
		const {
			size,
			buffer,
			buffable: { elementSize, elementRecover },
		} = this
		if (size.some((s) => s === 0)) {
			log.warn('Zero size buffer as output')
			return
		}
		const index: number[] = size.map(() => 0)
		let pos = 0
		if (elementRecover)
			do {
				yield [
					[...index] as InputSpec,
					// @ts-expect-error TODO: fix size
					elementRecover(buffer.subarray(pos, pos + elementSize), [
						/*todo*/
					]),
				]
				pos += elementSize
			} while (nextXdIndex(index, size))
		else
			do {
				yield [[...index] as InputSpec, buffer.subarray(pos, pos + elementSize) as OriginElement]
				pos += elementSize
			} while (nextXdIndex(index, size))
	}
	*values(): Generator<OriginElement> {
		yield* this.entries().map(([_, v]) => v)
	}
	toArray(): OriginElement[] {
		return [...this.values()]
	}
	// TODO: type
	slice(
		...index: [number, ...number[]]
	): BufferReader<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec> {
		const { size, buffer, buffable: specification } = this
		const { elementSize } = specification
		if (!this.size.length)
			throw new ArraySizeValidationError(`Cannot slice a ${this.size.length}-D buffer`)
		if (index.length > this.size.length)
			throw new ArraySizeValidationError(
				`Index length mismatch: taking ${index.length}-D index slice out of ${size.length}-D buffer`
			)
		const subSize = size.slice(index.length)
		const subLength = subSize.reduce((a, b) => a * b, 1)
		const pos = bufferPosition(index, size, elementSize) * subLength
		return new BufferReader<Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>(
			specification,
			buffer.subarray(pos, pos + subLength) as Buffer,
			subSize
		)
	}
	*slices() {
		const { size } = this
		if (!size.length) throw new ArraySizeValidationError(`Cannot slice a ${size.length}-D buffer`)
		for (let i = 0; i < size[0]; i++) yield this.slice(i)
	}
}
