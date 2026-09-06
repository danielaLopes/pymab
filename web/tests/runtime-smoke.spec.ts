import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

import { policyCatalog, policyIds } from "../src/catalog/policies";

test("home is accessible and mission links are present", async ({ page }) => {
  await page.goto("./#/");
  await expect(
    page.getByRole("heading", { name: /See how bandit algorithms choose/ }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "ε ε-greedy EpsilonGreedyPolicy" })).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
  await page.getByRole("button", { name: "Motion: system" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-reduced-motion", "true");
});

test("every policy route renders the shared history board", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "Complete route inventory runs once in Chromium");
  test.setTimeout(120_000);
  for (const policyId of policyIds) {
    await test.step(policyId, async () => {
      await page.goto(`./#/lesson/${policyId}`);
      await expect(page.locator(".current-run strong")).toContainText(
        policyCatalog[policyId].label,
        { timeout: 30_000 },
      );
      await expect(page.getByRole("region", { name: "Decision history" })).toBeVisible({
        timeout: 30_000,
      });
      await expect(page.getByRole("columnheader", { name: /Moon Path/ })).toBeVisible();
    });
  }
});

test("real PyMAB wheel completes a seeded epsilon decision", async ({ page }) => {
  await page.goto("./#/lesson/epsilon-greedy");
  const advance = page.getByRole("button", { name: "Advance one round" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await advance.click();
  await expect(page.getByText("Round 1: Relic found")).toBeVisible();
  await expect(page.getByRole("cell", { name: /Round 1, Star Path, chosen/ })).toBeVisible();
  await page.getByRole("button", { name: /Inspect PyMAB/ }).click();
  await expect(page.getByText("2.0.0", { exact: true })).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("the debrief free play shortcut keeps the active parameter", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "Long debrief scenario runs once in Chromium");
  await page.goto("./#/lesson/linucb");
  await expect(page.getByRole("button", { name: "Auto-run" })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole("button", { name: "Auto-run" }).click();
  await expect(page.getByRole("heading", { name: "Run complete" })).toBeVisible({
    timeout: 30_000,
  });
  await page.getByRole("button", { name: "Start free play" }).click();
  await expect(page.locator(".current-run strong")).toHaveText(
    "LinUCB · Free play · alpha 1 · l2 1 · seed 31415",
  );
  await expect(page.getByLabel("Random seed")).toBeEditable();
});

test("real LinUCB decision displays context and score decomposition", async ({ page }) => {
  await page.goto("./#/lesson/linucb");
  const advance = page.getByRole("button", { name: "Advance one round" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await advance.click();
  await expect(page.getByRole("row", { name: /Round 1.*Signals: light/ })).toBeVisible();
  await expect(page.getByRole("cell", { name: /UCB score/ })).toHaveCount(3);
  await page.getByRole("button", { name: "About light" }).hover();
  await expect(page.getByRole("tooltip")).toContainText(
    "Light can be red or blue. The policy sees it before choosing a path.",
  );
  await page.getByRole("button", { name: /Inspect PyMAB/ }).click();
  await expect(page.getByRole("table", { name: "LinUCB score decomposition" })).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("Python Lab runs PyMAB and reports output", async ({ page }) => {
  await page.goto("./#/lab");
  await page.getByRole("button", { name: "Run Python" }).click();
  await expect(page.getByText("Run complete.")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "stdout" })).toBeVisible();
  await expect(page.locator(".console-panel pre").filter({ hasText: "estimates:" })).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("completed run reveals environment values and the full regret path", async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Long debrief scenario runs once in Chromium");
  await page.goto("./#/lesson/epsilon-greedy");
  const autoRun = page.getByRole("button", { name: "Auto-run" });
  await expect(autoRun).toBeEnabled({ timeout: 30_000 });
  await autoRun.click();
  await expect(page.getByRole("heading", { name: "Run complete" })).toBeVisible({
    timeout: 30_000,
  });
  await page.getByText("Show environment values and regret by round").click();
  await expect(page.locator(".truth-grid div").filter({ hasText: "Star Path" })).toContainText(
    "75% reward chance",
  );
  await expect(page.locator(".truth-grid div").filter({ hasText: "Moon Path" })).toContainText(
    "Not selected in this run",
  );
  await expect(
    page.getByRole("table", { name: "Decision and expected-regret path" }),
  ).toBeVisible();
  await expect(
    page.getByRole("table", { name: "Decision and expected-regret path" }).getByRole("row"),
  ).toHaveCount(13);
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
  await page.getByRole("button", { name: "Start challenge" }).click();
  await expect(page.locator(".current-run strong")).toContainText("Challenge");
  await expect(page.getByRole("progressbar", { name: "Run progress" })).toHaveAttribute(
    "aria-valuenow",
    "0",
  );
});

test("draft changes preserve progress until the user starts a new run", async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Interaction scenario runs once in Chromium");
  await page.goto("./#/lesson/epsilon-greedy");
  const advance = page.getByRole("button", { name: "Advance one round" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await advance.click();
  await expect(page.getByText("1 / 12")).toBeVisible();
  await page.getByRole("radio", { name: "Challenge" }).click();
  await expect(page.getByText("Changes have not been applied.")).toBeVisible();
  await expect(page.locator(".current-run strong")).toContainText("Guided");
  await expect(page.getByText("1 / 12")).toBeVisible();

  await page.getByRole("button", { name: "Restart with these settings" }).click();
  await expect(page.locator(".current-run strong")).toContainText("Challenge");
  await expect(page.getByRole("progressbar", { name: "Run progress" })).toHaveAttribute(
    "aria-valuenow",
    "0",
  );
});

test("epsilon challenge can be completed by auto-run", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "Long resilience scenarios run once in Chromium");
  await page.goto("./#/lesson/epsilon-greedy");
  const challenge = page.getByRole("radio", { name: "Challenge" });
  await expect(challenge).toBeEnabled({ timeout: 30_000 });
  await challenge.click();
  await page.getByRole("button", { name: "Restart with these settings" }).click();
  const autoRun = page.getByRole("button", { name: "Auto-run" });
  await expect(autoRun).toBeEnabled();
  await autoRun.click();
  await expect(page.getByRole("heading", { name: "Challenge cleared" })).toBeVisible({
    timeout: 30_000,
  });
});

test("free play accepts exact parameters and an editable seed", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "Interaction scenario runs once in Chromium");
  await page.goto("./#/lesson/epsilon-greedy");
  await expect(page.getByRole("radio", { name: "Free play" })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole("radio", { name: "Free play" }).click();
  await page.getByRole("spinbutton", { name: "Exploration chance" }).fill("0.35");
  await page.getByLabel("Random seed").fill("1234");
  await expect(page.getByText("Generated from seed")).toBeVisible();
  const moonChance = page.getByRole("spinbutton", { name: "Moon" });
  await moonChance.fill("12.3");
  await expect(page.getByText("Custom")).toBeVisible();
  await page.getByRole("button", { name: "Regenerate from seed" }).click();
  await expect(page.getByText("Generated from seed")).toBeVisible();
  await expect(moonChance).not.toHaveValue("12.3");
  await moonChance.fill("12.3");
  await page.getByLabel("Random seed").fill("5678");
  await expect(moonChance).toHaveValue("12.3");
  await page.getByRole("button", { name: "Restart with these settings" }).click();
  await expect(page.locator(".current-run strong")).toHaveText(
    "ε-greedy · Free play · epsilon 0.35 · initial_value 0 · seed 5678",
  );
  await expect(page.getByRole("columnheader", { name: /Moon Path/ })).toContainText(
    "12.3% reward chance",
  );
});

test("policy selection starts the new policy immediately and carries the mode", async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Interaction scenario runs once in Chromium");
  await page.goto("./#/lesson/epsilon-greedy");
  await expect(page.getByRole("combobox", { name: "Policy" })).toBeEnabled({
    timeout: 30_000,
  });
  await page.getByRole("radio", { name: "Free play" }).click();
  await page.getByLabel("Random seed").fill("9876");
  await page.getByRole("combobox", { name: "Policy" }).click();
  await page.getByRole("option", { name: "LinUCB" }).click();
  await expect(page).toHaveURL(/#\/lesson\/linucb$/);
  await expect(page.getByRole("heading", { name: "The Labyrinth of Signals" })).toBeVisible();
  await expect(page.locator(".mission-header .eyebrow")).toHaveText(
    "Contextual bandits · Lin UCB Policy",
  );
  await expect(page.getByText("LinUCBPolicy", { exact: true })).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: "Confidence width" })).toHaveValue("1");
  await expect(page.getByLabel("Random seed")).toHaveValue("31415");
  await expect(page.locator(".current-run strong")).toContainText("LinUCB · Free play");
  await page.getByRole("button", { name: "Advance one round" }).click();
  await expect(page).toHaveURL(/#\/lesson\/linucb$/);
  await expect(page.locator(".current-run strong")).toContainText("LinUCB · Free play");

  await page.getByRole("combobox", { name: "Policy" }).click();
  await page.getByRole("option", { name: "ε-greedy", exact: true }).click();
  await expect(page).toHaveURL(/#\/lesson\/epsilon-greedy$/);
  await expect(page.getByRole("heading", { name: "The Three Ancient Gates" })).toBeVisible();
  await expect(page.locator(".current-run strong")).toContainText("ε-greedy · Free play");
});

test("challenge starts even when four attempts are already recorded", async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Persistence scenario runs once in Chromium");
  await page.addInitScript(() => {
    window.localStorage.setItem(
      "pymab-arcade:v1",
      JSON.stringify({
        version: 1,
        completed: ["epsilon-greedy"],
        attempts: { "epsilon-greedy": 4, linucb: 0 },
        preferences: { inspectorOpen: false, reducedMotion: null },
        recent: {
          "epsilon-greedy": { seed: 42, parameter: 0.2 },
          linucb: { seed: 31415, parameter: 1 },
        },
      }),
    );
  });
  await page.goto("./#/lesson/epsilon-greedy");
  await expect(page.getByRole("radio", { name: "Challenge" })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole("radio", { name: "Challenge" }).click();
  await page.getByRole("button", { name: "Restart with these settings" }).click();
  await expect(page.locator(".current-run strong")).toContainText("Challenge");
  await expect(page.getByRole("button", { name: "Advance one round" })).toBeEnabled();
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
  await expect(page.getByRole("button", { name: "Advance one round" })).toBeEnabled({
    timeout: 30_000,
  });
  const started = Date.now();
  await page.goto("./#/lesson/linucb");
  await expect(page.getByRole("button", { name: "Advance one round" })).toBeEnabled({
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
  await expect(page.getByRole("button", { name: "Advance one round" })).toBeEnabled({
    timeout: 30_000,
  });
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(overflow).toBe(false);
});

test("applied recommendation scenario learns from click feedback", async ({ page }) => {
  await page.goto("./#/scenario/recommendations");
  const advance = page.getByRole("button", { name: "Advance one round" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await expect(page.getByRole("columnheader", { name: /Article/ })).toBeVisible();
  await advance.click();
  await expect(page.getByRole("row", { name: /Round 1.*Signals: visitor/ })).toBeVisible();
  await expect(page.getByRole("cell", { name: /Predicted click probability/ })).toHaveCount(3);
  await expect(page.getByText(/Round 1: (Click|No click)/)).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("defensive verification exposes utility and keeps its safety boundary visible", async ({
  page,
}) => {
  await page.goto("./#/scenario/defensive-verification");
  const advance = page.getByRole("button", { name: "Advance one round" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await expect(page.getByText("Safety boundary", { exact: true })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: /Strong verification/ })).toBeVisible();
  await advance.click();
  await expect(page.getByRole("row", { name: /Round 1.*Signals: risk/ })).toBeVisible();
  await expect(page.getByText(/utility\)/)).toBeVisible();
  await page.getByRole("button", { name: /Inspect PyMAB/ }).click();
  await expect(page.getByRole("table", { name: "LinUCB score decomposition" })).toBeVisible();
});

test("scenario selection and free-play controls remain on the chosen route", async ({ page }) => {
  await page.goto("./#/scenario/recommendations");
  await expect(page.getByRole("combobox", { name: "Scenario" })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole("combobox", { name: "Scenario" }).click();
  await page.getByRole("option", { name: "Defensive verification" }).click();
  await expect(page).toHaveURL(/#\/scenario\/defensive-verification$/);
  await expect(page.getByRole("button", { name: "Advance one round" })).toBeEnabled();
  await page.getByRole("radio", { name: "Free play" }).click();
  await expect(page.getByLabel("Random seed")).toBeEditable();
  await page.getByLabel("Random seed").fill("808");
  await page.getByRole("button", { name: "Start configured run" }).click();
  await page.getByRole("button", { name: "Advance one round" }).click();
  await expect(page).toHaveURL(/#\/scenario\/defensive-verification$/);
});

test("scenario home and lesson avoid page-level overflow at 320 pixels", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 760 });
  for (const route of ["./#/", "./#/scenario/recommendations"] as const) {
    await page.goto(route);
    if (route.includes("scenario")) {
      await expect(page.getByRole("button", { name: "Advance one round" })).toBeEnabled({
        timeout: 30_000,
      });
    }
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
      ),
    ).toBe(false);
  }
});
