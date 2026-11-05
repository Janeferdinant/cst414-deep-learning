/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#070B19',
          900: '#0B1226',
          800: '#111D3E',
          700: '#1B2C5B',
        },
        burgundy: {
          900: '#3D070F',
          800: '#580A16',
          700: '#750F1F',
          600: '#96152B',
          500: '#B81D39',
        },
        gold: {
          100: '#FDF7E7',
          200: '#F9ECC7',
          300: '#F4DD9E',
          400: '#ECC86C',
          500: '#D4AF37', // Champagne / Metallic Gold
          600: '#B59124',
        }
      },
      fontFamily: {
        serif: ['Cinzel', 'Playfair Display', 'Georgia', 'serif'],
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}

