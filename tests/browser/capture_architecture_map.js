const {chromium} = require("playwright");
const path = require("path");


async function main() {
  const projectRoot = path.resolve(__dirname, "..", "..");
  const architectureRoot = path.join(projectRoot, "docs", "architecture");
  const browser = await chromium.launch({
    executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    headless: true,
  });

  try {
    const outputs = [];
    for (const item of [
      {name: "paperstorm-executive-overview", source: "svg", width: 1920, height: 1080},
      {name: "paperstorm-agent-system-flow", source: "svg", width: 2200, height: 1300},
      {name: "paperstorm-executive-overview-v57", source: "svg", width: 1920, height: 1080},
      {name: "paperstorm-agent-system-flow-v57", source: "svg", width: 2200, height: 1300},
    ]) {
      const sourcePath = path.join(architectureRoot, `${item.name}.${item.source}`);
      const outputPath = path.join(architectureRoot, `${item.name}.png`);
      const sourceUrl = `file:///${sourcePath.replace(/\\/g, "/")}`;
      const page = await browser.newPage({
        viewport: {width: item.width, height: item.height},
        deviceScaleFactor: 1,
      });
      await page.goto(sourceUrl, {waitUntil: "load"});
      await page.evaluate(() => document.fonts.ready);
      const dimensions = await page.locator("svg").evaluate((svg) => {
        const rect = svg.getBoundingClientRect();
        return [Math.round(rect.width), Math.round(rect.height)];
      });
      if (dimensions[0] !== item.width || dimensions[1] !== item.height) {
        throw new Error(`Unexpected ${item.name} dimensions: ${dimensions.join("x")}`);
      }
      await page.locator("svg").screenshot({path: outputPath, animations: "disabled"});
      outputs.push({name: item.name, dimensions, outputPath});
      await page.close();
    }

    const sourcePath = path.join(architectureRoot, "paperstorm-system-architecture.html");
    const outputPath = path.join(architectureRoot, "paperstorm-system-architecture.png");
    const sourceUrl = `file:///${sourcePath.replace(/\\/g, "/")}`;
    const page = await browser.newPage({viewport: {width: 2400, height: 1500}, deviceScaleFactor: 1});
    await page.goto(sourceUrl, {waitUntil: "load"});
    await page.evaluate(() => document.fonts.ready);

    const diagnostics = await page.evaluate(() => {
      const map = document.querySelector(".architecture-map");
      const mapRect = map.getBoundingClientRect();
      const overflow = [...document.querySelectorAll(".node, .panel")]
        .filter((element) => (
          element.scrollWidth > element.clientWidth + 1
          || element.scrollHeight > element.clientHeight + 1
        ))
        .map((element) => ({
          label: element.textContent.trim().replace(/\s+/g, " ").slice(0, 90),
          client: [element.clientWidth, element.clientHeight],
          scroll: [element.scrollWidth, element.scrollHeight],
        }));
      const outside = [...document.querySelectorAll(".panel, .node")]
        .filter((element) => {
          const rect = element.getBoundingClientRect();
          return rect.left < mapRect.left || rect.top < mapRect.top
            || rect.right > mapRect.right || rect.bottom > mapRect.bottom;
        })
        .map((element) => element.textContent.trim().replace(/\s+/g, " ").slice(0, 90));
      return {
        dimensions: [Math.round(mapRect.width), Math.round(mapRect.height)],
        overflow,
        outside,
      };
    });

    if (diagnostics.dimensions[0] !== 2400 || diagnostics.dimensions[1] !== 1500) {
      throw new Error(`Unexpected map dimensions: ${diagnostics.dimensions.join("x")}`);
    }
    if (diagnostics.overflow.length || diagnostics.outside.length) {
      throw new Error(`Layout diagnostics failed:\n${JSON.stringify(diagnostics, null, 2)}`);
    }

    await page.locator(".architecture-map").screenshot({
      path: outputPath,
      animations: "disabled",
    });
    outputs.push({name: "paperstorm-system-architecture", ...diagnostics, outputPath});
    process.stdout.write(`${JSON.stringify(outputs, null, 2)}\n`);
  } finally {
    await browser.close();
  }
}


main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
