/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const ports = Array.from({ length: 19 }, (_, i) => ({
      source: `/api/project/${i + 1}/:path*`,
      destination: `${process.env[`PROJECT_${i + 1}_URL`] || `http://localhost:${8000 + i + 1}`}/:path*`,
    }));
    return ports;
  },
};

module.exports = nextConfig;
