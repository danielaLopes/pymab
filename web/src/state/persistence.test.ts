import { defaultPersistedState, loadPersistence, savePersistence } from "./persistence";

class MemoryStorage implements Storage {
  private values = new Map<string, string>();
  get length() {
    return this.values.size;
  }
  clear() {
    this.values.clear();
  }
  getItem(key: string) {
    return this.values.get(key) ?? null;
  }
  key(index: number) {
    return [...this.values.keys()][index] ?? null;
  }
  removeItem(key: string) {
    this.values.delete(key);
  }
  setItem(key: string, value: string) {
    this.values.set(key, value);
  }
}

describe("lesson persistence", () => {
  it("round-trips only the versioned public preferences", () => {
    const storage = new MemoryStorage();
    const value = { ...defaultPersistedState, completed: ["epsilon-greedy" as const] };
    savePersistence(value, storage);
    expect(loadPersistence(storage)).toEqual(value);
  });

  it("discards corrupt and future data", () => {
    const storage = new MemoryStorage();
    storage.setItem("pymab-arcade:v1", "not-json");
    expect(loadPersistence(storage)).toEqual(defaultPersistedState);
    storage.setItem("pymab-arcade:v1", JSON.stringify({ version: 2 }));
    expect(loadPersistence(storage)).toEqual(defaultPersistedState);
  });

  it("migrates the two original policy records into version 2", () => {
    const storage = new MemoryStorage();
    storage.setItem(
      "pymab-arcade:v1",
      JSON.stringify({
        version: 1,
        completed: ["epsilon-greedy"],
        attempts: { "epsilon-greedy": 2, linucb: 1 },
        preferences: { inspectorOpen: true, reducedMotion: null },
        recent: {
          "epsilon-greedy": { seed: 91, parameter: 0.14 },
          linucb: { seed: 92, parameter: 1.3 },
        },
      }),
    );
    const migrated = loadPersistence(storage);
    expect(migrated.version).toBe(2);
    expect(migrated.completed).toEqual(["epsilon-greedy"]);
    expect(migrated.recent["epsilon-greedy"].parameters).toEqual({
      initial_value: 0,
      epsilon: 0.14,
    });
    expect(migrated.recent.linucb.parameters).toEqual({ alpha: 1.3, l2: 1 });
    expect(Object.keys(migrated.recent)).toHaveLength(27);
  });
});
