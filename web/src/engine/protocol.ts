import { z } from "zod";

import { policyIds } from "@/catalog/policies";

export const policyIdSchema = z.enum(policyIds);
export const scenarioIds = ["recommendations", "defensive-verification"] as const;
export const scenarioIdSchema = z.enum(scenarioIds);
export const lessonModeSchema = z.enum(["guided", "challenge", "freePlay"]);
export const policyFamilySchema = z.enum([
  "foundations",
  "optimism",
  "bayesian",
  "changing",
  "best-arm",
  "adversarial",
  "contextual",
]);
export const policyObjectiveSchema = z.enum(["cumulative-reward", "best-arm"]);

const parameterValueSchema = z.union([z.number(), z.boolean(), z.string(), z.null()]);

const requestBase = z.object({ requestId: z.string().min(1) });
const sessionBase = requestBase.extend({ sessionId: z.string().min(1) });
const environmentSchema = z.record(z.string(), z.unknown());

export const requestSchema = z.discriminatedUnion("type", [
  requestBase.extend({ type: z.literal("initialize"), sourceCommit: z.string().optional() }),
  sessionBase.extend({
    type: z.literal("startLesson"),
    policyId: policyIdSchema,
    mode: lessonModeSchema,
    seed: z.number().int(),
    parameters: z.record(z.string(), parameterValueSchema),
    environment: environmentSchema.optional(),
    sourceCommit: z.string().optional(),
  }),
  sessionBase.extend({
    type: z.literal("startScenario"),
    scenarioId: scenarioIdSchema,
    mode: lessonModeSchema,
    seed: z.number().int(),
    parameters: z.record(z.string(), parameterValueSchema),
    environment: environmentSchema.optional(),
    sourceCommit: z.string().optional(),
  }),
  sessionBase.extend({ type: z.literal("step") }),
  sessionBase.extend({ type: z.literal("runToEnd") }),
  sessionBase.extend({ type: z.literal("reset") }),
  sessionBase.extend({ type: z.literal("dispose") }),
]);

const cueSchema = z.object({
  name: z.string(),
  value: z.union([z.number(), z.string()]),
  label: z.string(),
});
const armPresentationSchema = z.object({
  id: z.string().min(1).optional(),
  name: z.string(),
  shortName: z.string(),
  symbolKind: z.enum([
    "moon",
    "sun",
    "star",
    "article",
    "product",
    "tutorial",
    "video",
    "podcast",
    "newsletter",
    "course",
    "event",
    "tool",
    "message",
    "offer",
    "download",
    "allow",
    "light-check",
    "strong-verification",
  ]),
});
const presentationSchema = z.object({
  experienceKind: z.enum(["policy", "scenario"]),
  experienceId: z.string(),
  arms: z.array(armPresentationSchema).min(2).max(8),
  rewardPresentation: z.enum(["binary", "numeric", "utility"]),
  positiveOutcomeLabel: z.string(),
  zeroOutcomeLabel: z.string(),
  contextFeatures: z
    .array(
      z.object({
        id: z.string(),
        name: z.string(),
        type: z.enum(["base", "binary", "numeric"]),
      }),
    )
    .max(9)
    .optional(),
});
const historyEventSchema = z.object({
  selectedArm: z.number().int().min(0).max(7),
  reward: z.number(),
  instantaneousExpectedRegret: z.number().nonnegative(),
  visibleCues: z.array(cueSchema),
  publicContext: z.array(z.array(z.number())).nullable(),
  explanationKey: z.string(),
  diagnostic: z.record(z.string(), z.unknown()),
});

const lessonSnapshotBaseSchema = z.object({
  policyId: policyIdSchema,
  scenarioId: scenarioIdSchema.nullable(),
  family: policyFamilySchema,
  objective: policyObjectiveSchema,
  mode: lessonModeSchema,
  seed: z.number().int(),
  packageVersion: z.string(),
  sourceCommit: z.string(),
  sessionId: z.string(),
  step: z.number().int().nonnegative(),
  horizon: z.number().int().positive(),
  parameters: z.record(z.string(), parameterValueSchema),
  environment: environmentSchema.nullable(),
  gateIds: z.array(z.string()).min(2).max(8),
  presentation: presentationSchema,
  selectedArm: z.number().int().min(0).max(7).nullable(),
  reward: z.number().nullable(),
  totalReward: z.number(),
  instantaneousExpectedRegret: z.number().nonnegative().nullable(),
  cumulativeExpectedRegret: z.number().nonnegative(),
  completed: z.boolean(),
  passed: z.boolean(),
  visibleCues: z.array(cueSchema),
  publicContext: z.array(z.array(z.number())).nullable(),
  explanationKey: z.string(),
  diagnostic: z.record(z.string(), z.unknown()).nullable(),
  recommendation: z.number().int().min(0).max(7).nullable(),
  history: z.array(historyEventSchema),
  hiddenTruth: z.record(z.string(), z.unknown()).nullable(),
  generatedCode: z.string(),
});

export const lessonSnapshotSchema = lessonSnapshotBaseSchema.superRefine((snapshot, context) => {
  const armCount = snapshot.presentation.arms.length;
  if (snapshot.gateIds.length !== armCount) {
    context.addIssue({
      code: "custom",
      path: ["gateIds"],
      message: "gateIds must match the presentation arm count",
    });
  }
  for (const [path, selected] of [
    [["selectedArm"], snapshot.selectedArm],
    [["recommendation"], snapshot.recommendation],
  ] as const) {
    if (selected !== null && selected >= armCount) {
      context.addIssue({
        code: "custom",
        path: [...path],
        message: "arm index is outside the presentation arm count",
      });
    }
  }
  snapshot.history.forEach((event, index) => {
    if (event.selectedArm >= armCount) {
      context.addIssue({
        code: "custom",
        path: ["history", index, "selectedArm"],
        message: "arm index is outside the presentation arm count",
      });
    }
  });
});

const responseBase = z.object({ requestId: z.string() });
const snapshotResponse = {
  sessionId: z.string(),
  snapshot: lessonSnapshotSchema,
};
export const responseSchema = z.discriminatedUnion("type", [
  responseBase.extend({
    type: z.literal("ready"),
    packageVersion: z.string(),
    sourceCommit: z.string(),
  }),
  responseBase.extend({ type: z.literal("lessonStarted"), ...snapshotResponse }),
  responseBase.extend({ type: z.literal("stepCompleted"), ...snapshotResponse }),
  responseBase.extend({ type: z.literal("runCompleted"), ...snapshotResponse }),
  responseBase.extend({ type: z.literal("disposed"), sessionId: z.string() }),
  responseBase.extend({
    type: z.literal("error"),
    error: z.object({
      code: z.enum([
        "BOOT_FAILED",
        "INVALID_REQUEST",
        "INVALID_SESSION",
        "POLICY_FAILED",
        "STALE_RESPONSE",
        "LAB_SYNTAX",
        "LAB_RUNTIME",
        "LAB_TIMEOUT",
        "OUTPUT_LIMIT",
      ]),
      message: z.string(),
      recoverable: z.boolean(),
      details: z.string().nullable().optional(),
    }),
  }),
]);

export const progressSchema = z.object({
  type: z.literal("progress"),
  stage: z.enum(["runtime", "numpy", "scipy", "pymab", "lesson"]),
  message: z.string(),
});

export type PolicyId = z.infer<typeof policyIdSchema>;
export type ScenarioId = z.infer<typeof scenarioIdSchema>;
export type LessonId = PolicyId;
export type LessonMode = z.infer<typeof lessonModeSchema>;
export type LessonRequest = z.infer<typeof requestSchema>;
export type LessonResponse = z.infer<typeof responseSchema>;
export type LessonSnapshot = z.infer<typeof lessonSnapshotSchema>;
export type RuntimeProgress = z.infer<typeof progressSchema>;
