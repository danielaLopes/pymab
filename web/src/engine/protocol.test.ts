import { requestSchema } from "./protocol";

describe("scenario protocol", () => {
  it("accepts a supported scenario request", () => {
    expect(
      requestSchema.parse({
        type: "startScenario",
        requestId: "request",
        sessionId: "session",
        scenarioId: "recommendations",
        mode: "guided",
        seed: 2401,
        parameters: { epsilon: 0.08 },
      }),
    ).toMatchObject({ type: "startScenario", scenarioId: "recommendations" });
  });

  it("rejects scenario names outside the dedicated registry", () => {
    expect(() =>
      requestSchema.parse({
        type: "startScenario",
        requestId: "request",
        sessionId: "session",
        scenarioId: "made-up",
        mode: "guided",
        seed: 1,
        parameters: {},
      }),
    ).toThrow();
  });
});
