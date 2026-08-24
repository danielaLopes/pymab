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
  await expect(page.getByRole("heading", { name: /Learn the art/ })).toBeVisible();
  await expect(page).toHaveScreenshot("campaign-map.png", screenshotOptions);
});

test("epsilon chamber and inspector", async ({ page }) => {
  await page.goto("./#/lesson/epsilon-greedy");
  const advance = page.getByRole("button", { name: "Advance one chamber" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await advance.click();
  await expect(page.getByText(/Chamber 1:/)).toBeVisible();
  await expect(page).toHaveScreenshot("epsilon-chamber.png", screenshotOptions);
  await page.getByRole("button", { name: /Inspect PyMAB/ }).click();
  await expect(page.getByRole("heading", { name: "Decision state" })).toBeVisible();
  await expect(page).toHaveScreenshot("epsilon-inspector.png", screenshotOptions);
});

test("LinUCB contextual chamber and inspector", async ({ page }) => {
  await page.goto("./#/lesson/linucb");
  const advance = page.getByRole("button", { name: "Advance one chamber" });
  await expect(advance).toBeEnabled({ timeout: 30_000 });
  await advance.click();
  await expect(page.getByRole("list", { name: "Current chamber signals" })).toBeVisible();
  await expect(page).toHaveScreenshot("linucb-chamber.png", screenshotOptions);
  await page.getByRole("button", { name: /Inspect PyMAB/ }).click();
  await expect(page.getByRole("table", { name: "LinUCB score decomposition" })).toBeVisible();
  await expect(page).toHaveScreenshot("linucb-inspector.png", screenshotOptions);
});

test("both guided debriefs", async ({ page }) => {
  for (const lesson of ["epsilon-greedy", "linucb"] as const) {
    await page.goto(`./#/lesson/${lesson}`);
    const autoRun = page.getByRole("button", { name: "Auto-run" });
    await expect(autoRun).toBeEnabled({ timeout: 30_000 });
    await autoRun.click();
    await expect(page.getByRole("heading", { name: "The map has learned from you" })).toBeVisible({
      timeout: 30_000,
    });
    await expect(page).toHaveScreenshot(`${lesson}-debrief.png`, screenshotOptions);
  }
});
