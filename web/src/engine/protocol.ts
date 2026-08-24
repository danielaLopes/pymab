import { z } from "zod";

export const lessonIdSchema = z.enum(["epsilon-greedy", "linucb"]);
export const lessonModeSchema = z.enum(["guided", "challenge", "freePlay"]);

const requestBase = z.object({ requestId: z.string().min(1) });
const sessionBase = requestBase.extend({ sessionId: z.string().min(1) });

export const requestSchema = z.discriminatedUnion("type", [
  requestBase.extend({ type: z.literal("initialize"), sourceCommit: z.string().optional() }),
  sessionBase.extend({
    type: z.literal("startLesson"),
    lessonId: lessonIdSchema,
    mode: lessonModeSchema,
    seed: z.number().int(),
    parameters: z.record(z.string(), z.number()),
    sourceCommit: z.string().optional(),
  }),
  sessionBase.extend({ type: z.literal("step") }),
  sessionBase.extend({ type: z.literal("runToEnd") }),
  sessionBase.extend({ type: z.literal("reset") }),
  sessionBase.extend({ type: z.literal("dispose") }),
]);

const cueSchema = z.object({
  name: z.string(),
  value: z.union([z.literal(-1), z.literal(1)]),
  label: z.string(),
});
const historyEventSchema = z.object({
  selectedArm: z.number().int().min(0).max(2),
  reward: z.number(),
  instantaneousExpectedRegret: z.number().nonnegative(),
  visibleCues: z.array(cueSchema),
  publicContext: z.array(z.array(z.number())).nullable(),
  explanationKey: z.string(),
  diagnostic: z.record(z.string(), z.unknown()),
});

export const lessonSnapshotSchema = z.object({
  lessonId: lessonIdSchema,
  mode: lessonModeSchema,
  seed: z.number().int(),
  packageVersion: z.string(),
  sourceCommit: z.string(),
  sessionId: z.string(),
  step: z.number().int().nonnegative(),
  horizon: z.number().int().positive(),
  parameters: z.record(z.string(), z.number()),
  gateIds: z.array(z.string()).length(3),
  selectedArm: z.number().int().min(0).max(2).nullable(),
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
  history: z.array(historyEventSchema),
  hiddenTruth: z.record(z.string(), z.unknown()).nullable(),
  generatedCode: z.string(),
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
  stage: z.enum(["runtime", "numpy", "pymab", "lesson"]),
  message: z.string(),
});

export type LessonId = z.infer<typeof lessonIdSchema>;
export type LessonMode = z.infer<typeof lessonModeSchema>;
export type LessonRequest = z.infer<typeof requestSchema>;
export type LessonResponse = z.infer<typeof responseSchema>;
export type LessonSnapshot = z.infer<typeof lessonSnapshotSchema>;
export type RuntimeProgress = z.infer<typeof progressSchema>;
