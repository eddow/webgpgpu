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
				.input({ a: vec2f.array(2) })
				.output({ output: vec2f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
					a: [
						[1, 2],
						[10, 20],
					],
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([[11, 22]])
		})
		it('vec3', async () => {
			const kernel = webGpGpu
				.input({ a: vec3f.array(2) })
				.output({ output: vec3f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
					a: [
						[1, 2, 3],
						[10, 20, 30],
					],
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([[11, 22, 33]])
		})
		it('vec4', async () => {
			const kernel = webGpGpu
				.input({ a: vec4f.array(2) })
				.output({ output: vec4f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
					a: [
						[1, 2, 3, 4],
						[10, 20, 30, 40],
					],
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([[11, 22, 33, 44]])
		})
	})
	describe('mat2x', () => {
		it('mat2x2', async () => {
			const kernel = webGpGpu
				.input({ a: mat2x2f.array(2) })
				.output({ output: mat2x2f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
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
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([
				[
					[11, 22],
					[55, 66],
				],
			])
		})
		it('mat2x3', async () => {
			const kernel = webGpGpu
				.input({ a: mat2x3f.array(2) })
				.output({ output: mat2x3f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
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
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([
				[
					[11, 22, 33],
					[55, 66, 77],
				],
			])
		})
		it('mat2x4', async () => {
			const kernel = webGpGpu
				.input({ a: mat2x4f.array(2) })
				.output({ output: mat2x4f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
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
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([
				[
					[11, 22, 33, 44],
					[55, 66, 77, 88],
				],
			])
		})
	})
	describe('mat3x', () => {
		it('mat3x2', async () => {
			const kernel = webGpGpu
				.input({ a: mat3x2f.array(2) })
				.output({ output: mat3x2f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
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
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([
				[
					[11, 22],
					[55, 66],
					[99, 110],
				],
			])
		})
		it('mat3x3', async () => {
			const kernel = webGpGpu
				.input({ a: mat3x3f.array(2) })
				.output({ output: mat3x3f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
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
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([
				[
					[11, 22, 33],
					[55, 66, 77],
					[99, 110, 121],
				],
			])
		})
		it('mat3x4', async () => {
			const kernel = webGpGpu
				.input({ a: mat3x4f.array(2) })
				.output({ output: mat3x4f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
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
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([
				[
					[11, 22, 33, 44],
					[55, 66, 77, 88],
					[99, 110, 121, 132],
				],
			])
		})
	})
	describe('mat4x', () => {
		it('mat4x2', async () => {
			const kernel = webGpGpu
				.input({ a: mat4x2f.array(2) })
				.output({ output: mat4x2f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
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
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([
				[
					[11, 22],
					[55, 66],
					[99, 110],
					[143, 154],
				],
			])
		})
		it('mat4x3', async () => {
			const kernel = webGpGpu
				.input({ a: mat4x3f.array(2) })
				.output({ output: mat4x3f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
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
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([
				[
					[11, 22, 33],
					[55, 66, 77],
					[99, 110, 121],
					[143, 154, 165],
				],
			])
		})
		it('mat4x4', async () => {
			const kernel = webGpGpu
				.input({ a: mat4x4f.array(2) })
				.output({ output: mat4x4f.array('threads.x') })
				.kernel('output[0] = a[0]+a[1];')
			const { output } = await kernel(
				{
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
				},
				{ 'threads.x': 1 }
			)
			expect(output.values()).to.typedArrayEqual([
				[
					[11, 22, 33, 44],
					[55, 66, 77, 88],
					[99, 110, 121, 132],
					[143, 154, 165, 176],
				],
			])
		})
	})
})
