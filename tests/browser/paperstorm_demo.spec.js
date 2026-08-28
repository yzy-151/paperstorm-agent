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
  await expect(page.locator(".release-pill strong")).toHaveText("v7.1");
  await expect(page.locator("#developer-view")).toBeHidden();

  await page.locator("#show-research-mode").click();
  await expect(page.locator(".workspace-inspector")).toBeHidden();
  const deliveryNode = page.locator('.pipeline-node[data-node="deliver"]');
  await expect(deliveryNode).not.toHaveClass(/complete/);
  await expect(page.locator("#research-score-section")).toBeHidden();
  await expect(page.locator("#article-content")).toContainText("完成调研后");
  await page.screenshot({
    path: path.join(screenshotRoot, "research-before.png"),
    fullPage: true,
  });

  await page.locator("#task-run-mode").selectOption("fake", {force: true});
  await page.locator("#start-research-demo").click();
  await expect(page.locator("#start-research-demo")).toBeEnabled({timeout: 30000});
  await expect(page.locator("#research-current-activity")).toContainText("调研完成");
  await expect(page.locator("#article-content")).not.toBeEmpty();
  await expect(deliveryNode).toHaveClass(/complete/);
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
  await expect(page.locator(".benchmark-card")).toHaveCount(9, {timeout: 30000});
  await expect(page.locator("#ready-root-detail")).toContainText("paperstorm-benchmarks");

  const pimPilot = page.locator('[data-benchmark-id="pim-domain-pilot"]');
  await expect(pimPilot).toContainText("PIM");

  const longMemory = page.locator('[data-benchmark-id="longmemeval-retrieval"]');
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

test("chat exposes session list, expandable citations and regenerate without losing history", async ({page}) => {
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(baseURL, {waitUntil: "networkidle"});
  await page.locator("#show-chat-mode").click();
  await page.locator("#chat-run-mode").selectOption("fake");

  const sessionsBefore = await page.locator(".session-item").count().catch(() => 0);
  await page.locator("#create-chat").click();
  await expect(page.locator(".session-item").first()).toBeVisible({timeout: 10000});
  await page.locator("#chat-input").fill("PIM 是什么，神经网络如何抑制它？");
  await page.locator("#send-chat").click();

  const lastMessage = page.locator("#chat-messages .message.assistant").last();
  await expect(lastMessage).toContainText(/神经网络|PIM|无源互调/, {timeout: 60000});
  await expect(lastMessage.locator(".citations summary")).toContainText("引用");
  await lastMessage.locator(".citations summary").click();
  await expect(lastMessage.locator(".citations li").first()).toBeVisible();
  const sessionsAfter = await page.locator(".session-item").count();
  expect(sessionsAfter).toBeGreaterThanOrEqual(sessionsBefore + 1);

  await page.locator("#regenerate-chat").click();
  await expect(page.locator("#chat-messages .message").last()).toContainText("v2", {timeout: 60000});
  await expect(page.locator("#chat-messages .message").last()).toContainText("重新生成");
  expect(await page.locator("#stop-chat").count()).toBe(1);
  await expect(page.locator("#stop-chat")).toBeHidden();

  await page.screenshot({
    path: path.join(screenshotRoot, "chat-session-citations.png"),
    fullPage: true,
  });
});

test("research article downloads as markdown and developer console stays clean", async ({page}) => {
  await page.setViewportSize({width: 1366, height: 768});
  await page.goto(baseURL, {waitUntil: "networkidle"});
  await page.locator("#task-run-mode").selectOption("fake", {force: true});

  const downloadPromise = page.waitForEvent("download", {timeout: 15000});
  await page.locator("#start-research-demo").click();
  await expect(page.locator("#article-content")).not.toBeEmpty({timeout: 60000});
  await expect(page.locator("#download-article-md")).toBeEnabled();
  await page.locator("#download-article-md").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.md$/);

  await page.locator("#show-developer-mode").click();
  await expect(page.locator(".benchmark-card")).toHaveCount(9, {timeout: 30000});
  await expect(page.locator("#leave-developer-mode")).toBeVisible();
});
