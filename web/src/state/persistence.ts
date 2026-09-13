import { z } from "zod";

import { policyCatalog, policyIds, type ParameterValues, type PolicyId } from "@/catalog/policies";

const STORAGE_KEY = "pymab-arcade:v1";
const policyIdSchema = z.enum(policyIds);
const parameterValueSchema = z.union([z.number(), z.boolean(), z.string(), z.null()]);
const persistedSchema = z.object({
  version: z.literal(2),
  completed: z.array(policyIdSchema),
  attempts: z.record(policyIdSchema, z.number().int().nonnegative()),
  preferences: z.object({ inspectorOpen: z.boolean(), reducedMotion: z.boolean().nullable() }),
  recent: z.record(
    policyIdSchema,
    z.object({
      seed: z.number().int(),
      parameters: z.record(z.string(), parameterValueSchema),
    }),
  ),
});

const legacySchema = z.object({
  version: z.literal(1),
  completed: z.array(z.enum(["epsilon-greedy", "linucb"])),
  attempts: z.record(z.enum(["epsilon-greedy", "linucb"]), z.number().int().nonnegative()),
  preferences: z.object({ inspectorOpen: z.boolean(), reducedMotion: z.boolean().nullable() }),
  recent: z.record(
    z.enum(["epsilon-greedy", "linucb"]),
    z.object({ seed: z.number().int(), parameter: z.number() }),
  ),
});

export type PersistedState = z.infer<typeof persistedSchema>;

function initialAttempts(): Record<PolicyId, number> {
  return Object.fromEntries(policyIds.map((id) => [id, 0])) as Record<PolicyId, number>;
}

function initialRecent(): Record<PolicyId, { seed: number; parameters: ParameterValues }> {
  return Object.fromEntries(
    policyIds.map((id) => [
      id,
      { seed: policyCatalog[id].guidedSeed, parameters: { ...policyCatalog[id].defaults } },
    ]),
  ) as Record<PolicyId, { seed: number; parameters: ParameterValues }>;
}

export const defaultPersistedState: PersistedState = {
  version: 2,
  completed: [],
  attempts: initialAttempts(),
  preferences: { inspectorOpen: false, reducedMotion: null },
  recent: initialRecent(),
};

function migrateLegacy(value: z.infer<typeof legacySchema>): PersistedState {
  const migrated = structuredClone(defaultPersistedState);
  migrated.completed = [...value.completed];
  migrated.preferences = value.preferences;
  for (const id of ["epsilon-greedy", "linucb"] as const) {
    migrated.attempts[id] = value.attempts[id];
    migrated.recent[id] = {
      seed: value.recent[id].seed,
      parameters:
        id === "epsilon-greedy"
          ? { initial_value: 0, epsilon: value.recent[id].parameter }
          : { alpha: value.recent[id].parameter, l2: 1 },
    };
  }
  return migrated;
}

export function loadPersistence(
  storage: Storage | undefined = globalThis.localStorage,
): PersistedState {
  if (!storage) return structuredClone(defaultPersistedState);
  try {
    const value = storage.getItem(STORAGE_KEY);
    if (!value) return structuredClone(defaultPersistedState);
    const decoded: unknown = JSON.parse(value);
    const current = persistedSchema.safeParse(decoded);
    if (current.success) return current.data;
    const legacy = legacySchema.safeParse(decoded);
    return legacy.success ? migrateLegacy(legacy.data) : structuredClone(defaultPersistedState);
  } catch {
    return structuredClone(defaultPersistedState);
  }
}

export function savePersistence(
  value: PersistedState,
  storage: Storage | undefined = globalThis.localStorage,
): void {
  if (!storage) return;
  storage.setItem(STORAGE_KEY, JSON.stringify(persistedSchema.parse(value)));
}
