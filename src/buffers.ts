import type { Buffable, NumericSizesSpec } from 'dataTypes'
import { log } from './system'
import {
	ArraySizeValidationError,
	type Input0D,
	type Input1D,
	type Input2D,
	type SizeSpec,
	type TypedArray,
	type WorkSizeInfer,
	assertElementSize,
	assertSize,
	inferSize,
	isTypedArrayXD,
} from './typedArrays'

export class SizeInferError extends Error {}

export function elementsToTypedArray<
	Buffer extends TypedArray,
	OriginElement,
	InputSizesSpec extends SizeSpec[],
>(
	specification: Buffable<Buffer, OriginElement, SizeSpec[], InputSizesSpec>,
	workSizeInfer: WorkSizeInfer,
	data: any,
	size: SizeSpec[]
): Buffer {
	function inferredSizes<SS extends SizeSpec[]>(size: SS) {
		return size.map((s) => {
			if (typeof s !== 'number') throw new SizeInferError('Size should have been inferred - TODO')
			return s
		}) as NumericSizesSpec<SS>
	}
	const { bufferType, elementSize, elementConvert } = specification
	// #region 0D
	if (size.length === 0) {
		if (data instanceof bufferType) {
			assertElementSize((data as Buffer).length, elementSize)
			return data as Buffer
		}
		const elm = elementConvert
			? elementConvert(data as OriginElement, inferredSizes(specification.transformSize))
			: data
		return elm instanceof bufferType ? elm : new bufferType(elm)
	}
	// #endregion 0D
	// #region 1D
	if (size.length === 1) {
		if (data instanceof bufferType) {
			if ((data as Buffer).length % elementSize !== 0)
				throw new ArraySizeValidationError(
					`Size mismatch in dimension 1: ${data.length} is not a multiple of ${elementSize}`
				)
			assertSize([(data as Buffer).length / elementSize], [size[0]], workSizeInfer)
			return data as Buffer
		}
		if (!Array.isArray(data))
			throw new ArraySizeValidationError('Input is not an array nor a typed array')
		assertSize([data.length], [size[0]], workSizeInfer)
		const rv = new bufferType(data.length * elementSize) as Buffer
		let dst = 0
		// Make the `if` early not to not make it in the loop
		if (elementConvert) {
			for (const element of data as OriginElement[]) {
				rv.set(elementConvert(element, inferredSizes(specification.transformSize)), dst)
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
		assertSize(data.size, size, workSizeInfer)
		return data as Buffer
	}
	if (!Array.isArray(data))
		throw new ArraySizeValidationError('Input is not an array nor a typed array')
	assertSize([data.length], size.slice(0, 1), workSizeInfer)
	let rv: Buffer | undefined
	let itemSize: number | undefined
	let dst = 0
	for (const element of data as (
		| Input0D<OriginElement>
		| Input1D<OriginElement>
		| Input2D<OriginElement>
	)[]) {
		const subBuffer = elementsToTypedArray(specification, workSizeInfer, element, size.slice(1))
		if (!rv) {
			itemSize = subBuffer.length
			// TODO: multidimensional inferring
			rv = new bufferType(inferSize(size, workSizeInfer))
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
	for (let i = index.length - 1; i > 0; i--) {
		index[i]++
		if (index[i] < size[i]) return true
		index[i] = 0
	}
	return false
}

export class BufferReader<Buffer extends TypedArray, OriginElement> {
	constructor(
		public readonly specification: Buffable<Buffer, OriginElement, SizeSpec[]>,
		public readonly buffer: Buffer,
		public readonly size: number[]
	) {}
	element(...index: number[]): OriginElement {
		const {
			size,
			buffer,
			specification: { elementSize, elementRecover },
		} = this
		if (index.length !== size.length)
			throw new ArraySizeValidationError(
				`Index length mismatch: taking ${index.length}-D index element out of ${size.length}-D buffer`
			)
		const pos = bufferPosition(index, size, elementSize)
		const element = buffer.subarray(pos, pos + elementSize)
		return (elementRecover?.(element, [
			/*todo*/
		]) ?? element) as OriginElement
	}
	*elements(): Generator<[number[], OriginElement]> {
		const {
			size,
			buffer,
			specification: { elementSize, elementRecover },
		} = this
		const index: number[] = size.map(() => 0)
		let pos = 0
		if (elementRecover)
			while (nextXdIndex(index, size)) {
				yield [
					[...index],
					elementRecover(buffer.subarray(pos, pos + elementSize), [
						/*todo*/
					]),
				]
				pos += elementSize
			}
		else
			while (nextXdIndex(index, size)) {
				yield [[...index], buffer.subarray(pos, pos + elementSize) as OriginElement]
				pos += elementSize
			}
	}
	slice(...index: [number, ...number[]]): BufferReader<Buffer, OriginElement> {
		const { size, buffer, specification } = this
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
		return new BufferReader<Buffer, OriginElement>(
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
