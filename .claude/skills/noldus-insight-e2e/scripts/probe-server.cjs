// Probe whether local make dev is up on E2E_BASE_URL (default :2026).
// Exit 0 = up; exit 1 = down. Uses node http (no playwright dep needed here).
const { config } = require('./lib');
const http = require('http');
const https = require('https');
const url = require('url');

const { BASE_URL } = config();
const u = url.parse(BASE_URL);
const lib = u.protocol === 'https:' ? https : http;

const req = lib.request(
  { hostname: u.hostname, port: u.port, path: '/', method: 'GET', timeout: 5000 },
  (res) => {
    // 200/307/302 all mean the server is alive (307 = redirect to /login or /workspace).
    res.resume();
    if (res.statusCode && res.statusCode < 500) {
      console.log(`UP ${res.statusCode} ${BASE_URL}`);
      process.exit(0);
    } else {
      console.log(`UNHEALTHY ${res.statusCode} ${BASE_URL}`);
      process.exit(1);
    }
  },
);
req.on('error', (e) => { console.log(`DOWN ${e.message} (${BASE_URL})`); process.exit(1); });
req.on('timeout', () => { console.log(`DOWN timeout (${BASE_URL})`); req.destroy(); process.exit(1); });
req.end();
