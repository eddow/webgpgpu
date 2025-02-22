import createWebGpGpu, { f32, threads, u32, type WebGpGpu } from '../src/server'

let webGpGpu: WebGpGpu

function cpu(n: number) {
	const result = new Float32Array(n)
	for (let i = 0; i < n; i++) result[i] = i * i
	return Promise.resolve(result)
}
async function benchmark(fct: (n: number) => Promise<unknown>, n: number) {
	performance.mark(`start ${fct.name} ${n}`)
	await fct(1 << n)
	performance.mark(`end ${fct.name} ${n}`)
	return performance.measure(fct.name, `start ${fct.name} ${n}`, `end ${fct.name} ${n}`).duration
}
async function main() {
	webGpGpu = await createWebGpGpu()
	const gpuSquaresKernel = webGpGpu
		.output({ output: u32.array(threads.x) })
		.kernel('output[thread.x] = thread.x*thread.x;')
	function gpu(n: number) {
		return gpuSquaresKernel({}, { x: n })
	}
	//await gpu(65537)
	console.log(webGpGpu.device.limits.maxBufferSize / 4)
	/*for (let exp = 8; exp <= 30; exp++) {
		console.log(exp, ':', await benchmark(gpu, exp), '|', await benchmark(cpu, exp))
	}*/
	webGpGpu.dispose()
}

main()
