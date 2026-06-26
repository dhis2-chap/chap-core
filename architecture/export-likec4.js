// Screenshot each LikeC4 view from a running static build (`likec4 build` + served).
// LikeC4's own `export png` fails headless in Docker ("Failed N of N views"), so we
// render the static site and screenshot it instead. Driven by `make architecture-export-likec4`.
const { chromium } = require("playwright");

const BASE_URL = process.env.LIKEC4_URL || "http://localhost:5180";
const OUTPUT_DIR = process.env.OUTPUT_DIR || "/out";

// LikeC4 view id -> output filename (kept aligned with the Structurizr view keys).
const VIEWS = [
  ["index", "L1_Landscape"],
  ["chapCore", "L2_ChapCore"],
  ["chapkit", "L2_Chapkit"],
  ["coreApi", "L3_CoreAPI"],
  ["coreWorker", "L3_CoreWorker"],
  ["chapkitService", "L3_ChapkitService"],
];

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1680, height: 1050 } });
  for (const [id, name] of VIEWS) {
    await page.goto(`${BASE_URL}/view/${id}/`, { waitUntil: "networkidle" });
    await page.waitForTimeout(2500); // let the layout settle / fit to view
    await page.screenshot({ path: `${OUTPUT_DIR}/${name}.png` });
    console.log(`  ${name}.png`);
  }
  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
