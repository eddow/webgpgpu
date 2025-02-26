import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import typescript from '@rollup/plugin-typescript'

export default  [
	{
		output: {
			format: 'iife',
			sourcemap: true,
			name: 'webgpgpu-test',
		},
		plugins: [
			resolve(),
			commonjs({
				include: 'node_modules/**',
				ignore: ['js-base64'],
				sourceMap: false,
				namedExports: {
					chai: ['expect']
				}
			}),
			typescript({
				tsconfig: './tests/tsconfig.json',
				paths: {
					webgpgpu: ["./src/client/index.ts"]
				}
			})
		],
		onwarn: function(warning) {
			/*if (warning.code === 'CIRCULAR_DEPENDENCY') {
				return;
			}*/
			console.warn(`(!) ${warning.message}`);
		}
	}
]