import { expect } from 'chai'
import createWebGpGpu, {
	mat2x2f,
	mat2x3f,
	mat2x4f,
	mat3x2f,
	mat3x3f,
	mat3x4f,
	mat4x2f,
	mat4x3f,
	mat4x4f,
	vec2f,
	vec3f,
	vec4f,
	type RootWebGpGpu,
} from 'webgpgpu'
import './chai-ext'

describe('atomic', () => {
	let webGpGpu: RootWebGpGpu

	before(async () => {
		webGpGpu = await createWebGpGpu()
	})
	after(() => {
		webGpGpu.dispose()
	})
	describe('vec', () => {
		it('vec2', async () => {
			const kernel = webGpGpu
				.input({ a: vec2f.array('threads.x'), b: vec2f.array('threads.x') })
				.output({ output: vec2f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')
			const { output } = await kernel({
				a: [
					[1, 2],
					[10, 20],
				],
				b: [
					[3, 4],
					[30, 40],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[4, 6],
				[40, 60],
			])
		})
		it('vec3', async () => {
			const kernel = webGpGpu
				.input({ a: vec3f.array('threads.x'), b: vec3f.array('threads.x') })
				.output({ output: vec3f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')
			const { output } = await kernel({
				a: [
					[1, 2, 3],
					[10, 20, 30],
				],
				b: [
					[3, 4, 5],
					[30, 40, 50],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[4, 6, 8],
				[40, 60, 80],
			])
		})
		it('vec3*', async () => {
			const kernel = webGpGpu
				.input({ a: vec3f.array('threads.x'), b: vec3f.array('threads.x') })
				.output({ output: vec3f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]*b[thread.x];')
			const { output } = await kernel({
				a: [
					[1, 2, 3],
					[10, 20, 30],
				],
				b: [
					[3, 4, 5],
					[30, 40, 50],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[3, 8, 15],
				[300, 800, 1500],
			])
		})
		it('vec4', async () => {
			const kernel = webGpGpu
				.input({ a: vec4f.array('threads.x'), b: vec4f.array('threads.x') })
				.output({ output: vec4f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')
			const { output } = await kernel({
				a: [
					[1, 2, 3, 4],
					[10, 20, 30, 40],
				],
				b: [
					[3, 4, 5, 6],
					[30, 40, 50, 60],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[4, 6, 8, 10],
				[40, 60, 80, 100],
			])
		})
	})
	describe('mat2x', () => {
		it('mat2x2', async () => {
			const kernel = webGpGpu
				.input({ a: mat2x2f.array('threads.x'), b: mat2x2f.array('threads.x') })
				.output({ output: mat2x2f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')
			const { output } = await kernel({
				a: [
					[
						[1, 2],
						[5, 6],
					],
					[
						[10, 20],
						[50, 60],
					],
				],
				b: [
					[
						[3, 4],
						[7, 8],
					],
					[
						[30, 40],
						[70, 80],
					],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[
					[4, 6],
					[12, 14],
				],
				[
					[40, 60],
					[120, 140],
				],
			])
		})
		it('mat2x3', async () => {
			const kernel = webGpGpu
				.input({ a: mat2x3f.array('threads.x'), b: mat2x3f.array('threads.x') })
				.output({ output: mat2x3f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')
			const { output } = await kernel({
				a: [
					[
						[1, 2, 3],
						[5, 6, 7],
					],
					[
						[10, 20, 30],
						[50, 60, 70],
					],
				],
				b: [
					[
						[3, 4, 5],
						[7, 8, 9],
					],
					[
						[30, 40, 50],
						[70, 80, 90],
					],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[
					[4, 6, 8],
					[12, 14, 16],
				],
				[
					[40, 60, 80],
					[120, 140, 160],
				],
			])
		})
		it('mat2x4', async () => {
			const kernel = webGpGpu
				.input({ a: mat2x4f.array('threads.x'), b: mat2x4f.array('threads.x') })
				.output({ output: mat2x4f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')

			const { output } = await kernel({
				a: [
					[
						[1, 2, 3, 4],
						[5, 6, 7, 8],
					],
					[
						[10, 20, 30, 40],
						[50, 60, 70, 80],
					],
				],
				b: [
					[
						[3, 4, 5, 6],
						[7, 8, 9, 10],
					],
					[
						[30, 40, 50, 60],
						[70, 80, 90, 100],
					],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[
					[4, 6, 8, 10],
					[12, 14, 16, 18],
				],
				[
					[40, 60, 80, 100],
					[120, 140, 160, 180],
				],
			])
		})
	})
	describe('mat3x', () => {
		it('mat3x2', async () => {
			const kernel = webGpGpu
				.input({ a: mat3x2f.array('threads.x'), b: mat3x2f.array('threads.x') })
				.output({ output: mat3x2f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')

			const { output } = await kernel({
				a: [
					[
						[1, 2],
						[5, 6],
						[9, 10],
					],
					[
						[10, 20],
						[50, 60],
						[90, 100],
					],
				],
				b: [
					[
						[3, 4],
						[7, 8],
						[11, 12],
					],
					[
						[30, 40],
						[70, 80],
						[110, 120],
					],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[
					[4, 6],
					[12, 14],
					[20, 22],
				],
				[
					[40, 60],
					[120, 140],
					[200, 220],
				],
			])
		})

		it('mat3x3', async () => {
			const kernel = webGpGpu
				.input({ a: mat3x3f.array('threads.x'), b: mat3x3f.array('threads.x') })
				.output({ output: mat3x3f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')

			const { output } = await kernel({
				a: [
					[
						[1, 2, 3],
						[5, 6, 7],
						[9, 10, 11],
					],
					[
						[10, 20, 30],
						[50, 60, 70],
						[90, 100, 110],
					],
				],
				b: [
					[
						[3, 4, 5],
						[7, 8, 9],
						[11, 12, 13],
					],
					[
						[30, 40, 50],
						[70, 80, 90],
						[110, 120, 130],
					],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[
					[4, 6, 8],
					[12, 14, 16],
					[20, 22, 24],
				],
				[
					[40, 60, 80],
					[120, 140, 160],
					[200, 220, 240],
				],
			])
		})

		it('mat3x4', async () => {
			const kernel = webGpGpu
				.input({ a: mat3x4f.array('threads.x'), b: mat3x4f.array('threads.x') })
				.output({ output: mat3x4f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')

			const { output } = await kernel({
				a: [
					[
						[1, 2, 3, 4],
						[5, 6, 7, 8],
						[9, 10, 11, 12],
					],
					[
						[10, 20, 30, 40],
						[50, 60, 70, 80],
						[90, 100, 110, 120],
					],
				],
				b: [
					[
						[3, 4, 5, 6],
						[7, 8, 9, 10],
						[11, 12, 13, 14],
					],
					[
						[30, 40, 50, 60],
						[70, 80, 90, 100],
						[110, 120, 130, 140],
					],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[
					[4, 6, 8, 10],
					[12, 14, 16, 18],
					[20, 22, 24, 26],
				],
				[
					[40, 60, 80, 100],
					[120, 140, 160, 180],
					[200, 220, 240, 260],
				],
			])
		})
	})
	describe('mat4x', () => {
		it('mat4x2', async () => {
			const kernel = webGpGpu
				.input({ a: mat4x2f.array('threads.x'), b: mat4x2f.array('threads.x') })
				.output({ output: mat4x2f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')

			const { output } = await kernel({
				a: [
					[
						[1, 2],
						[5, 6],
						[9, 10],
						[13, 14],
					],
					[
						[10, 20],
						[50, 60],
						[90, 100],
						[130, 140],
					],
				],
				b: [
					[
						[3, 4],
						[7, 8],
						[11, 12],
						[15, 16],
					],
					[
						[30, 40],
						[70, 80],
						[110, 120],
						[150, 160],
					],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[
					[4, 6],
					[12, 14],
					[20, 22],
					[28, 30],
				],
				[
					[40, 60],
					[120, 140],
					[200, 220],
					[280, 300],
				],
			])
		})

		it('mat4x3', async () => {
			const kernel = webGpGpu
				.input({ a: mat4x3f.array('threads.x'), b: mat4x3f.array('threads.x') })
				.output({ output: mat4x3f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')

			const { output } = await kernel({
				a: [
					[
						[1, 2, 3],
						[5, 6, 7],
						[9, 10, 11],
						[13, 14, 15],
					],
					[
						[10, 20, 30],
						[50, 60, 70],
						[90, 100, 110],
						[130, 140, 150],
					],
				],
				b: [
					[
						[3, 4, 5],
						[7, 8, 9],
						[11, 12, 13],
						[15, 16, 17],
					],
					[
						[30, 40, 50],
						[70, 80, 90],
						[110, 120, 130],
						[150, 160, 170],
					],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[
					[4, 6, 8],
					[12, 14, 16],
					[20, 22, 24],
					[28, 30, 32],
				],
				[
					[40, 60, 80],
					[120, 140, 160],
					[200, 220, 240],
					[280, 300, 320],
				],
			])
		})

		it('mat4x4', async () => {
			const kernel = webGpGpu
				.input({ a: mat4x4f.array('threads.x'), b: mat4x4f.array('threads.x') })
				.output({ output: mat4x4f.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')

			const { output } = await kernel({
				a: [
					[
						[1, 2, 3, 4],
						[5, 6, 7, 8],
						[9, 10, 11, 12],
						[13, 14, 15, 16],
					],
					[
						[10, 20, 30, 40],
						[50, 60, 70, 80],
						[90, 100, 110, 120],
						[130, 140, 150, 160],
					],
				],
				b: [
					[
						[3, 4, 5, 6],
						[7, 8, 9, 10],
						[11, 12, 13, 14],
						[15, 16, 17, 18],
					],
					[
						[30, 40, 50, 60],
						[70, 80, 90, 100],
						[110, 120, 130, 140],
						[150, 160, 170, 180],
					],
				],
			})

			expect(output.flat()).to.typedArrayEqual([
				[
					[4, 6, 8, 10],
					[12, 14, 16, 18],
					[20, 22, 24, 26],
					[28, 30, 32, 34],
				],
				[
					[40, 60, 80, 100],
					[120, 140, 160, 180],
					[200, 220, 240, 260],
					[280, 300, 320, 340],
				],
			])
		})
	})
	/*describe('structs', () => {
		it('struct', async () => {
			const struct = new Struct({memA: f32, memB: vec2f})
			const kernel = webGpGpu
				.output({ output: struct.array('threads.x') })
				.kernel('output[thread.x] = a[thread.x]+b[thread.x];')
			const { output } = await kernel({
					
				}))
	})*/
})
