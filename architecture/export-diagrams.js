// Export every Structurizr view to a PNG using the diagram scripting API.
// Runs against a running Structurizr `local` instance (see `make architecture-export`).
// Not run directly: invoked inside the Playwright Docker image by the Makefile.
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const BASE_URL = process.env.STRUCTURIZR_URL || "http://host.docker.internal:8080";
const WORKSPACE_ID = process.env.STRUCTURIZR_WORKSPACE_ID || "1";
const OUTPUT_DIR = process.env.OUTPUT_DIR || "/work/diagrams";

async function main() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });

  const diagramsUrl = `${BASE_URL}/workspace/${WORKSPACE_ID}/diagrams`;
  await page.goto(diagramsUrl, { waitUntil: "networkidle" });

  // Wait until the scripting API is ready.
  await page.waitForFunction(
    () => window.structurizr && window.structurizr.scripting && typeof window.structurizr.scripting.getViews === "function",
    null,
    { timeout: 60000 },
  );

  const views = await page.evaluate(() => window.structurizr.scripting.getViews().map((v) => v.key));
  if (!views.length) throw new Error("No views found in workspace");
  console.log(`Exporting ${views.length} views to ${OUTPUT_DIR}`);

  for (const key of views) {
    const dataUri = await page.evaluate(
      (viewKey) =>
        new Promise((resolve, reject) => {
          const s = window.structurizr.scripting;
          s.changeView(viewKey);
          const started = Date.now();
          const poll = () => {
            if (s.isDiagramRendered && s.isDiagramRendered()) {
              s.exportCurrentDiagramToPNG({ crop: false }, (png) => resolve(png));
            } else if (Date.now() - started > 30000) {
              reject(new Error(`render timeout for ${viewKey}`));
            } else {
              setTimeout(poll, 250);
            }
          };
          poll();
        }),
      key,
    );

    const base64 = dataUri.replace(/^data:image\/png;base64,/, "");
    const file = path.join(OUTPUT_DIR, `${key}.png`);
    fs.writeFileSync(file, Buffer.from(base64, "base64"));
    console.log(`  ${key}.png`);
  }

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
