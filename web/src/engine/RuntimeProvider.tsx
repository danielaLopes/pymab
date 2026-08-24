import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import type { RuntimeProgress } from "./protocol";
import { WorkerClient } from "./WorkerClient";

interface RuntimeContextValue {
  client: WorkerClient;
  progress: RuntimeProgress | null;
}

const RuntimeContext = createContext<RuntimeContextValue | null>(null);

export function RuntimeProvider({ children }: { children: ReactNode }) {
  const client = useMemo(() => new WorkerClient(), []);
  const [progress, setProgress] = useState<RuntimeProgress | null>(null);

  useEffect(() => {
    const unsubscribe = client.onProgress(setProgress);
    return () => {
      unsubscribe();
      client.dispose();
    };
  }, [client]);

  return <RuntimeContext.Provider value={{ client, progress }}>{children}</RuntimeContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useRuntime(): RuntimeContextValue {
  const context = useContext(RuntimeContext);
  if (!context) throw new Error("useRuntime must be used inside RuntimeProvider");
  return context;
}
