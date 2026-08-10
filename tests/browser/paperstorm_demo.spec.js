const {test, expect} = require("@playwright/test");
const path = require("path");

const baseURL = process.env.PAPERSTORM_DEMO_URL || "http://127.0.0.1:8002";
const screenshotRoot = process.env.PAPERSTORM_SCREENSHOT_DIR || "test-results/paperstorm-demo";

test.use({
  launchOptions: {
    executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  },
});

test("desktop one-click demo completes and renders the research result", async ({page}) => {
  await page.setViewportSize({width: 1366, height: 768});
  await page.goto(baseURL, {waitUntil: "networkidle"});
  await expect(page.locator(".release-pill strong")).toHaveText("v5.6");
  await expect(page.locator("#developer-view")).toBeHidden();

  await page.locator("#show-research-mode").click();
  await expect(page.locator(".demo-strip")).toBeVisible();
  await expect(page.locator('[data-stage="completed"]')).not.toHaveClass(/complete/);
  await expect(page.locator("#research-score-section")).toBeHidden();
  await expect(page.locator("#article-content")).toContainText("完成调研后");
  await page.screenshot({
    path: path.join(screenshotRoot, "research-before.png"),
    fullPage: true,
  });

  await page.locator("#start-research-demo").click();
  await expect(page.locator("#start-research-demo")).toBeEnabled({timeout: 30000});
  await expect(page.locator("#research-current-activity")).toContainText("调研完成");
  await expect(page.locator("#article-content")).not.toBeEmpty();
  await expect(page.locator('[data-stage="completed"]')).toHaveClass(/complete/);
  await page.screenshot({
    path: path.join(screenshotRoot, "research-completed.png"),
    fullPage: true,
  });
});

test("mobile chat keeps the primary conversation controls visible", async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.goto(baseURL, {waitUntil: "networkidle"});
  await page.locator("#show-chat-mode").click();
  await expect(page.locator("#chat-messages")).toBeVisible();
  await expect(page.locator("#chat-input")).toBeVisible();
  await expect(page.locator("#send-chat")).toBeVisible();
  await expect(page.locator("#developer-view")).toBeHidden();
  const overflowing = await page.evaluate(() =>
    [...document.querySelectorAll("body *")]
      .filter(element => {
        const rect = element.getBoundingClientRect();
        return rect.right > window.innerWidth + 1 || rect.left < -1;
      })
      .map(element => ({
        tag: element.tagName,
        id: element.id,
        className: String(element.className || ""),
        right: Math.round(element.getBoundingClientRect().right),
        width: Math.round(element.getBoundingClientRect().width),
      }))
      .slice(0, 12)
  );
  expect(overflowing, JSON.stringify(overflowing, null, 2)).toEqual([]);
  await page.screenshot({
    path: path.join(screenshotRoot, "chat-mobile.png"),
    fullPage: true,
  });
});

test("developer workbench discovers local datasets and exposes reproducible runs", async ({page}) => {
  await page.setViewportSize({width: 1440, height: 1000});
  await page.goto(baseURL, {waitUntil: "networkidle"});
  await page.locator("#show-developer-mode").click();

  await expect(page.locator("#developer-view")).toBeVisible();
  await expect(page.locator(".benchmark-card")).toHaveCount(6, {timeout: 30000});
  await expect(page.locator("#ready-root-detail")).toContainText("paperstorm-benchmarks");

  const longMemory = page.locator('[data-benchmark-id="longmemeval-retrieval-v56"]');
  await expect(longMemory).toContainText("LongMemEval");
  await longMemory.click();
  await expect(page.locator("#benchmark-selected-name")).toContainText("LongMemEval");
  await expect(page.locator("#benchmark-input-manifest")).toContainText("READY");
  await expect(page.locator("#benchmark-command-preview")).toContainText("run_longmemeval_benchmark.py");
  await expect(page.locator("#start-benchmark-run")).toBeEnabled();

  await page.screenshot({
    path: path.join(screenshotRoot, "benchmark-workbench.png"),
    fullPage: true,
  });
});
