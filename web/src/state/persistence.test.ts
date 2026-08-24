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
});
