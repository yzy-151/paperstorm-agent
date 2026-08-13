const {chromium} = require("playwright");
const path = require("path");

const baseURL = process.env.PAPERSTORM_DEMO_URL || "http://127.0.0.1:8002";
const root = path.resolve(__dirname, "..", "..");
const output = path.join(root, "docs", "screenshots");

async function assertLayout(page, name) {
  const diagnostics = await page.evaluate(() => ({
    viewport: [window.innerWidth, window.innerHeight],
    document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
    horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
    emptyAssets: [...document.images].filter((img) => !img.complete || !img.naturalWidth).map((img) => img.src),
  }));
  if (diagnostics.horizontalOverflow || diagnostics.emptyAssets.length) {
    throw new Error(`${name} layout failed: ${JSON.stringify(diagnostics)}`);
  }
  return diagnostics;
}

async function main() {
  const browser = await chromium.launch({
    executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    headless: true,
  });
  const page = await browser.newPage({viewport: {width: 1600, height: 1000}, deviceScaleFactor: 1});
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });

  try {
    await page.goto(baseURL, {waitUntil: "networkidle"});
    await page.locator("#task-run-mode").selectOption("fake", {force: true});
    await page.locator("#start-research-demo").click();
    await page.locator("#start-research-demo").waitFor({state: "visible"});
    await page.waitForFunction(() => document.querySelector("#research-current-activity")?.textContent.includes("调研完成"));
    await page.locator("#task-run-mode").selectOption("paperstorm", {force: true});
    await page.screenshot({path: path.join(output, "dashboard-research-v57.png"), fullPage: true});
    const research = await assertLayout(page, "research");

    await page.locator("#show-chat-mode").click();
    await page.screenshot({path: path.join(output, "dashboard-chat-v57.png"), fullPage: true});
    const chat = await assertLayout(page, "chat");

    await page.locator("#show-developer-mode").click();
    await page.locator(".benchmark-card").first().waitFor();
    const target = page.locator('[data-benchmark-id="longmemeval-retrieval-v56"]');
    await target.click();
    await page.screenshot({path: path.join(output, "dashboard-developer-v57.png"), fullPage: true});
    const developer = await assertLayout(page, "developer");

    if (errors.length) throw new Error(`Browser errors: ${JSON.stringify(errors)}`);
    process.stdout.write(`${JSON.stringify({research, chat, developer, errors}, null, 2)}\n`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
