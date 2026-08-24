import { z } from "zod";

const STORAGE_KEY = "pymab-arcade:v1";
const persistedSchema = z.object({
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

export const defaultPersistedState: PersistedState = {
  version: 1,
  completed: [],
  attempts: { "epsilon-greedy": 0, linucb: 0 },
  preferences: { inspectorOpen: false, reducedMotion: null },
  recent: {
    "epsilon-greedy": { seed: 42, parameter: 0.2 },
    linucb: { seed: 31415, parameter: 1.0 },
  },
};

export function loadPersistence(
  storage: Storage | undefined = globalThis.localStorage,
): PersistedState {
  if (!storage) return structuredClone(defaultPersistedState);
  try {
    const value = storage.getItem(STORAGE_KEY);
    if (!value) return structuredClone(defaultPersistedState);
    const parsed = persistedSchema.safeParse(JSON.parse(value));
    return parsed.success ? parsed.data : structuredClone(defaultPersistedState);
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
