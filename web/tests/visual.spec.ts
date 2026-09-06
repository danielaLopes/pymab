import { expect, test } from "@playwright/test";

const screenshotOptions = {
  animations: "disabled" as const,
  maxDiffPixelRatio: 0.08,
  threshold: 0.3,
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => window.localStorage.clear());
  await page.emulateMedia({ reducedMotion: "reduce", colorScheme: "dark" });
});

test("campaign map", async ({ page }) => {
  await page.goto("./#/");
  await expect(
    page.getByRole("heading", { name: /See how bandit algorithms choose/ }),
  ).toBeVisible();
  await expect(page).toHaveScreenshot("campaign-map.png", screenshotOptions);
  await expect(page.locator(".applied-scenarios")).toHaveScreenshot(
    "applied-scenarios.png",
    screenshotOptions,
  );
});

test("applied contextual scenarios", async ({ page }) => {
  for (const scenario of ["recommendations", "defensive-verification"] as const) {
    await page.goto(`./#/scenario/${scenario}`);
    const advance = page.getByRole("button", { name: "Advance one round" });
    await expect(advance).toBeEnabled({ timeout: 30_000 });
    await advance.click();
    const board = page.getByRole("region", { name: "Decision history" });
    await expect(board.getByRole("row", { name: /Round 1\./ })).toBeVisible();
    await expect(board).toHaveScreenshot(`${scenario}-history-region.png`, screenshotOptions);
  }
});

test("eight recommendation candidates", async ({ page }) => {
  await page.goto("./#/scenario/recommendations");
  await expect(page.getByRole("button", { name: "Advance one round" })).toBeEnabled({
    timeout: 30_000,
  });
  await page.getByRole("button", { name: /Run settings/ }).click();
  for (let index = 0; index < 5; index += 1) {
    await page.getByRole("button", { name: "Add tutorial" }).click();
  }
  await page.getByRole("button", { name: "Start configured run" }).click();
  await page.getByRole("button", { name: "Advance one round" }).click();
  await expect(page.getByRole("region", { name: "Decision history" })).toHaveScreenshot(
    "recommendations-eight-candidates.png",
    screenshotOptions,
  );
});

test("run setup panels", async ({ page }) => {
  for (const lesson of ["epsilon-greedy", "linucb"] as const) {
    await page.goto(`./#/lesson/${lesson}`);
    const panel = page.getByRole("region", { name: "Configure this run" });
    await expect(panel).toBeVisible({ timeout: 30_000 });
    await expect(panel).toHaveScreenshot(`${lesson}-run-setup.png`, screenshotOptions);
  }

  await page.goto("./#/lesson/epsilon-greedy");
  await expect(page.getByRole("radio", { name: "Free play" })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole("radio", { name: "Free play" }).click();
  await expect(page.getByText("Portal reward chances")).toBeVisible();
  await expect(page.getByRole("region", { name: "Configure this run" })).toHaveScreenshot(
    "epsilon-free-play-run-setup.png",
    screenshotOptions,
  );
});

test("epsilon round and inspector", async ({ page }) => {
  await page.goto("./#/lesson/epsilon-greedy");
  const advance = page.getByRole("button", { name: "Advance one round" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await advance.click();
  await expect(page.getByText(/Round 1:/)).toBeVisible();
  await expect(page).toHaveScreenshot("epsilon-history-board.png", screenshotOptions);
  await page.getByRole("button", { name: /Inspect PyMAB/ }).click();
  await expect(page.getByRole("heading", { name: "Decision state" })).toBeVisible();
  await expect(page).toHaveScreenshot("epsilon-inspector.png", screenshotOptions);
});

test("LinUCB contextual round and inspector", async ({ page }) => {
  await page.goto("./#/lesson/linucb");
  const advance = page.getByRole("button", { name: "Advance one round" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await advance.click();
  await expect(page.getByRole("row", { name: /Round 1.*Signals: light/ })).toBeVisible();
  await expect(page).toHaveScreenshot("linucb-history-board.png", screenshotOptions);
  await page.getByRole("button", { name: /Inspect PyMAB/ }).click();
  await expect(page.getByRole("table", { name: "LinUCB score decomposition" })).toBeVisible();
  await expect(page).toHaveScreenshot("linucb-inspector.png", screenshotOptions);
});

test("history board across policy families", async ({ page }) => {
  const policies = [
    "epsilon-greedy",
    "ucb",
    "gaussian-thompson-sampling",
    "sliding-window-ucb",
    "successive-elimination",
    "exp3",
    "linucb",
  ] as const;
  for (const policyId of policies) {
    await page.goto(`./#/lesson/${policyId}`);
    const advance = page.getByRole("button", { name: "Advance one round" });
    await expect(advance).toBeEnabled({ timeout: 30_000 });
    await advance.click();
    const board = page.getByRole("region", { name: "Decision history" });
    await expect(board.getByRole("row", { name: /Round 1\./ })).toBeVisible();
    await expect(board).toHaveScreenshot(`${policyId}-history-region.png`, screenshotOptions);
  }
});

test("both guided debriefs", async ({ page }) => {
  for (const lesson of ["epsilon-greedy", "linucb"] as const) {
    await page.goto(`./#/lesson/${lesson}`);
    const autoRun = page.getByRole("button", { name: "Auto-run" });
    await expect(autoRun).toBeEnabled({ timeout: 30_000 });
    await autoRun.click();
    await expect(page.getByRole("heading", { name: "Run complete" })).toBeVisible({
      timeout: 30_000,
    });
    await expect(page).toHaveScreenshot(`${lesson}-debrief.png`, screenshotOptions);
  }
});
