import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("home is accessible and mission links are present", async ({ page }) => {
  await page.goto("./#/");
  await expect(page.getByRole("heading", { name: /Learn the art of choosing/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /Three Ancient Gates/ })).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
  await page.getByRole("button", { name: "Motion: system" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-reduced-motion", "true");
});

test("real PyMAB wheel completes a seeded epsilon decision", async ({ page }) => {
  await page.goto("./#/lesson/epsilon-greedy");
  const advance = page.getByRole("button", { name: "Advance one chamber" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await advance.click();
  await expect(page.getByText("Chamber 1: Relic found")).toBeVisible();
  await expect(page.getByRole("button", { name: /Star Gate.*selected by PyMAB/ })).toBeVisible();
  await page.getByRole("button", { name: /Inspect PyMAB/ }).click();
  await expect(page.getByText("2.0.0", { exact: true })).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("real LinUCB decision displays context and score decomposition", async ({ page }) => {
  await page.goto("./#/lesson/linucb");
  const advance = page.getByRole("button", { name: "Advance one chamber" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await advance.click();
  await expect(page.getByRole("list", { name: "Current chamber signals" })).toBeVisible();
  await page.getByRole("button", { name: /Inspect PyMAB/ }).click();
  await expect(page.getByRole("table", { name: "LinUCB score decomposition" })).toBeVisible();
});

test("Python Lab runs PyMAB and reports output", async ({ page }) => {
  await page.goto("./#/lab");
  await page.getByRole("button", { name: "Run Python" }).click();
  await expect(page.getByText("Run complete.")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "stdout" })).toBeVisible();
  await expect(page.locator(".console-panel pre").filter({ hasText: "estimates:" })).toBeVisible();
});

test("epsilon challenge can be completed by auto-run", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "Long resilience scenarios run once in Chromium");
  await page.goto("./#/lesson/epsilon-greedy");
  const challenge = page.getByRole("button", { name: "Challenge" });
  await expect(challenge).toBeEnabled({ timeout: 30_000 });
  await challenge.click();
  const autoRun = page.getByRole("button", { name: "Auto-run" });
  await expect(autoRun).toBeEnabled();
  await autoRun.click();
  await expect(page.getByRole("heading", { name: "Challenge cleared" })).toBeVisible({
    timeout: 30_000,
  });
});

test("Python Lab reports syntax errors, times out, and recovers cleanly", async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Long resilience scenarios run once in Chromium");
  await page.goto("./#/lab");
  const editor = page.locator(".cm-content");
  await editor.fill("print(");
  await page.getByRole("button", { name: "Run Python" }).click();
  await expect(page.getByText("Run ended: syntax.")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".stderr")).toContainText("SyntaxError");

  await editor.fill('print("x" * 70000)');
  await page.getByRole("button", { name: "Run Python" }).click();
  await expect(page.getByText("Output was truncated at 64 KiB.")).toBeVisible({ timeout: 15_000 });

  await editor.fill("while True:\n    pass");
  await page.getByRole("button", { name: "Run Python" }).click();
  await expect(page.getByText("Run ended: timeout.")).toBeVisible({ timeout: 15_000 });
  await page.getByRole("button", { name: "ε-greedy example" }).click();
  await page.getByRole("button", { name: "Run Python" }).click();
  await expect(page.getByText("Run complete.")).toBeVisible({ timeout: 30_000 });

  await editor.fill("while True:\n    pass");
  await page.getByRole("button", { name: "Run Python" }).click();
  await page.waitForTimeout(200);
  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.getByRole("status")).toContainText(/stopped/i);
});

test("warm lesson switching stays responsive", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "Performance profile is calibrated for Chromium");
  await page.addInitScript(() => {
    const durations: number[] = [];
    Object.defineProperty(window, "__pymabLongTasks", { value: durations });
    new PerformanceObserver((list) => {
      durations.push(...list.getEntries().map((entry) => entry.duration));
    }).observe({ type: "longtask", buffered: true });
  });
  await page.goto("./#/lesson/epsilon-greedy");
  await expect(page.getByRole("button", { name: "Advance one chamber" })).toBeEnabled({
    timeout: 30_000,
  });
  const started = Date.now();
  await page.goto("./#/lesson/linucb");
  await expect(page.getByRole("button", { name: "Advance one chamber" })).toBeEnabled({
    timeout: 2_000,
  });
  expect(Date.now() - started).toBeLessThan(2_000);
  await expect(page.getByRole("heading", { name: "The Labyrinth of Signals" })).toBeFocused();
  const longestTask = await page.evaluate(() =>
    Math.max(0, ...((window as Window & { __pymabLongTasks?: number[] }).__pymabLongTasks ?? [])),
  );
  expect(longestTask).toBeLessThanOrEqual(100);
});

test("narrow layout has no horizontal overflow", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 760 });
  await page.goto("./#/lesson/epsilon-greedy");
  await expect(page.getByRole("button", { name: "Advance one chamber" })).toBeEnabled({
    timeout: 30_000,
  });
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(overflow).toBe(false);
});
