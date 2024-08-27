import flowbitePlugin from 'flowbite/plugin';
/** @type {import('tailwindcss').Config} */
export default {
	content: [
		'./src/**/*.{html,js,svelte,ts}',
		'./node_modules/flowbite-svelte/**/*.{html,js,svelte,ts}'
	],

	plugins: [flowbitePlugin],

	// darkMode: 'class',

	theme: {
		extend: {
			screens: {
				sm: '',
				xs: ''
			},
			fontSize: {},
			colors: {
				red: {
					50: '#ffe5e7',   // Very light red
					100: '#ffccd0',  // Light red
					200: '#ff99a1',  // Soft red
					300: '#ff6671',  // Moderate red
					400: '#ff3342',  // Bright red
					500: '#ff0012',  // Vivid red
					600: '#cc0010',  // Darker red
					700: '#a91d26',  // Base dark red (your specified color)
					800: '#80000b',  // Deeper red
					900: '#4d0006',  // Darkest red
					950: '#260003'   // Near black with a red tint
				}
			}
		}
	}
};
