import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import typescript from '@rollup/plugin-typescript'

export default function (config) {
	config.set({
		files: [
			// Watch src files for changes but
			// don't load them into the browser.
			//{ pattern: 'src/**/*.ts', included: false },
			{ pattern: 'tests/**/*.test.ts', type: 'module' }
		],
		browsers: ['Chrome'], // Or any other browser
		frameworks: ['mocha', 'chai'],
		rollupPreprocessor: {
			output: {
				format: 'iife', // Helps prevent naming collisions.
				name: 'webgpgpu-test', // Required for 'iife' format.
				sourcemap: 'inline', // Sensible for testing.
			},
			plugins: [
				resolve(),
				commonjs(),
				typescript({
					tsconfig: './tests/tsconfig.json',
					paths: {
						webgpgpu: ["./src/client/index.ts"]
					}
				})
			],
			//external: ['mocha', 'chai']
		},
		client: {
			mocha: {
				reporter: 'html'
			}
		}
	})
}