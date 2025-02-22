export default function (config) {
	config.set({
		frameworks: ['mocha'],
		files: [
      		'node_modules/mocha/mocha.js', // Prebuilt browser Mocha
			{pattern: 'tests/shims/process.js', type: 'module'},
			{pattern: 'tests/shims/**/*.js', type: 'module'},
			//'node_modules/stream-browserify/index.js', // Polyfill stream
			{pattern: 'tests/**/*.test.ts', type: 'module'},
		],
		preprocessors: {
			"tests/**/*.ts": ["esbuild"], // Transpile TypeScript to ESM
		},
		esbuild: {
			target: "esnext",
			format: "iife",
			bundle: true,
			external: ['util', 'events', 'stream', 'browser-stdout', './src/server/index.ts'],
			alias: {
				'browser-stdout': './tests/shims/browser-stdout.js'
			},
			define: {
				'process': 'globalThis.process', // ✅ Prevent process from being removed
				'process.env': '{}',  // ✅ Ensures Mocha doesn’t crash when checking process.env
			}
		},
		reporters: ['mocha'],
		port: 9876,  // karma web server port
		colors: true,
		logLevel: config.LOG_INFO,
		browsers: ['ChromeHeadless'],
		autoWatch: false,
		concurrency: Infinity,
		client: {
			mocha: {
				globals: ['mocha', 'chai'], // Expose Mocha and Chai globals
				setup: function () {
					mocha.setup('bdd'); // Initialize Mocha
				},
			},
		},
	})
}