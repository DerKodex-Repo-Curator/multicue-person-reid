/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#090d0c",
        panel: "#111715",
        line: "#28312e",
        mint: "#55e6a5",
        amber: "#f4bf63",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(85,230,165,.15), 0 24px 80px rgba(0,0,0,.35)",
      },
    },
  },
  plugins: [],
};
