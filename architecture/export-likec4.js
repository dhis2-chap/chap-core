// Export each LikeC4 view to a clean, chrome-free PNG.
// LikeC4's own headless `export png` fails in Docker ("Failed N of N views") and its
// in-app PNG export does not trigger a capturable download headless, so we render the
// served static build, hide the LikeC4 UI overlays (nav, Share/Export, attribution,
// node action buttons), and screenshot just the diagram canvas.
// Driven by `make architecture-export-likec4`.
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
  ["flowIngestDataset", "Flow_IngestDataset"],
  ["flowBacktest", "Flow_Backtest"],
  ["flowPrediction", "Flow_Prediction"],
];

// Hide everything that is not the diagram itself. `.likec4-root` has exactly two
// children: the `.react-flow` canvas and the overlay layer (nav / breadcrumb / search).
const HIDE_CSS = `
  .likec4-root > div:not(.react-flow),
  .likec4-navigation-panel__root,
  .react-flow__attribution,
  .react-flow__handle,
  [class*="details-button"],
  [class*="action-buttons"],
  [class*="action-btn"] { display: none !important; }
`;

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1680, height: 1050 }, colorScheme: "dark" });

  for (const [id, name] of VIEWS) {
    await page.goto(`${BASE_URL}/view/${id}/`, { waitUntil: "networkidle" });
    // Fail loudly if the view did not actually render. The SPA serves its shell
    // even for unknown routes (single-page mode), so a fixed wait alone could
    // silently screenshot an empty/error page after a bad id or route change.
    await page.waitForSelector(".react-flow__node", { timeout: 15000 });
    await page.addStyleTag({ content: HIDE_CSS });
    await page.waitForTimeout(2000); // let the layout settle / fit to view
    await page.screenshot({ path: `${OUTPUT_DIR}/${name}.png` });
    console.log(`  ${name}.png`);
  }

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
