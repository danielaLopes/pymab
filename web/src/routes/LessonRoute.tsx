import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import {
  Chamber,
  Debrief,
  ErrorRecovery,
  InspectPanel,
  LoadingStages,
  MissionHeader,
  ModeTabs,
  OutcomeReveal,
  ParameterChallenge,
  ProgressTrail,
  RunControls,
  UnsupportedBrowser,
} from "../components/game";
import { explanationCopy, lessonContent } from "../content/lessons";
import type { LessonId, LessonMode, LessonResponse, LessonSnapshot } from "../engine/protocol";
import { useRuntime } from "../engine/RuntimeProvider";
import { detectBrowserSupport } from "../engine/support";
import { WorkerClient } from "../engine/WorkerClient";
import { initialLessonState, lessonReducer } from "../state/lessonReducer";
import { loadPersistence, savePersistence } from "../state/persistence";

function snapshotFrom(response: LessonResponse): LessonSnapshot {
  if ("snapshot" in response) return response.snapshot;
  if (response.type === "error") throw new Error(response.error.message);
  throw new Error(`Unexpected worker response: ${response.type}`);
}

export function LessonRoute() {
  const { lessonSlug } = useParams();
  const lessonId: LessonId = lessonSlug === "linucb" ? "linucb" : "epsilon-greedy";
  const content = lessonContent[lessonId];
  const { client, progress } = useRuntime();
  const navigate = useNavigate();
  const [state, dispatch] = useReducer(lessonReducer, initialLessonState);
  const [persisted, setPersisted] = useState(() => loadPersistence());
  const [parameter, setParameter] = useState(lessonId === "epsilon-greedy" ? 0.2 : 1.0);
  const [seed, setSeed] = useState(content.guidedSeed);
  const [inspectorOpen, setInspectorOpen] = useState(persisted.preferences.inspectorOpen);
  const [autoRunning, setAutoRunning] = useState(false);
  const autoRef = useRef(false);
  const sessionRef = useRef("");
  const lessonRef = useRef(lessonId);
  const parameterRef = useRef(parameter);
  const seedRef = useRef(seed);
  const persistedRef = useRef(persisted);
  const recordedSessionRef = useRef("");

  const startMode = useCallback(
    async (mode: LessonMode, nextParameter?: number, nextSeed?: number) => {
      autoRef.current = false;
      setAutoRunning(false);
      dispatch({ type: "pending", value: true });
      try {
        if (mode === "challenge" && persistedRef.current.attempts[lessonId] >= 3) {
          throw new Error(
            "All three challenge attempts are complete. Free play is still available.",
          );
        }
        if (sessionRef.current) {
          await client.send({
            type: "dispose",
            requestId: WorkerClient.requestId(),
            sessionId: sessionRef.current,
          });
        }
        const sessionId = WorkerClient.requestId();
        sessionRef.current = sessionId;
        const resolvedParameter = nextParameter ?? parameterRef.current;
        const resolvedSeed =
          nextSeed ??
          (mode === "guided"
            ? content.guidedSeed
            : mode === "challenge"
              ? content.challengeSeed
              : seedRef.current);
        setSeed(resolvedSeed);
        seedRef.current = resolvedSeed;
        const response = await client.send({
          type: "startLesson",
          requestId: WorkerClient.requestId(),
          sessionId,
          lessonId,
          mode,
          seed: resolvedSeed,
          parameters:
            lessonId === "epsilon-greedy"
              ? { epsilon: resolvedParameter }
              : { alpha: resolvedParameter, l2: 1 },
        });
        dispatch({ type: "started", mode, snapshot: snapshotFrom(response) });
      } catch (error) {
        dispatch({
          type: "error",
          message: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [client, content.challengeSeed, content.guidedSeed, lessonId],
  );

  useEffect(() => {
    if (lessonRef.current !== lessonId) {
      lessonRef.current = lessonId;
      const defaultParameter = lessonId === "epsilon-greedy" ? 0.2 : 1.0;
      setParameter(defaultParameter);
      parameterRef.current = defaultParameter;
    }
    const support = detectBrowserSupport();
    if (!support.supported) {
      dispatch({
        type: "unsupported",
        message: support.reason ?? "Required browser features are unavailable.",
      });
      return;
    }
    let active = true;
    void client
      .initialize()
      .then(() => {
        if (active) {
          return startMode("guided", lessonId === "epsilon-greedy" ? 0.2 : 1.0, content.guidedSeed);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          dispatch({
            type: "error",
            message: error instanceof Error ? error.message : String(error),
          });
        }
      });
    return () => {
      active = false;
      autoRef.current = false;
    };
  }, [client, content.guidedSeed, lessonId, startMode]);

  useEffect(
    () => () => {
      autoRef.current = false;
      const sessionId = sessionRef.current;
      sessionRef.current = "";
      if (!sessionId) return;
      void client
        .send({
          type: "dispose",
          requestId: WorkerClient.requestId(),
          sessionId,
        })
        .catch(() => client.restart());
    },
    [client],
  );

  const advance = useCallback(async (): Promise<LessonSnapshot | null> => {
    if (!sessionRef.current) return null;
    dispatch({ type: "pending", value: true });
    try {
      const response = await client.send({
        type: "step",
        requestId: WorkerClient.requestId(),
        sessionId: sessionRef.current,
      });
      const snapshot = snapshotFrom(response);
      dispatch({ type: "snapshot", snapshot });
      return snapshot;
    } catch (error) {
      dispatch({ type: "error", message: error instanceof Error ? error.message : String(error) });
      return null;
    }
  }, [client]);

  const autoRun = useCallback(() => {
    autoRef.current = true;
    setAutoRunning(true);
    void (async () => {
      while (autoRef.current) {
        if (document.hidden) {
          autoRef.current = false;
          break;
        }
        const snapshot = await advance();
        if (!snapshot || snapshot.completed) break;
        await new Promise((resolve) => window.setTimeout(resolve, 360));
      }
      setAutoRunning(false);
    })();
  }, [advance]);

  const changeMode = (mode: LessonMode) => {
    if (
      state.snapshot?.step &&
      !window.confirm("Start a fresh expedition? Current progress will reset.")
    )
      return;
    void startMode(mode);
  };

  const changeParameter = (value: number) => {
    if (
      state.snapshot?.step &&
      !window.confirm("Changing this parameter starts a fresh expedition.")
    )
      return;
    setParameter(value);
    parameterRef.current = value;
    const next = {
      ...persistedRef.current,
      recent: {
        ...persistedRef.current.recent,
        [lessonId]: { seed: seedRef.current, parameter: value },
      },
    };
    persistedRef.current = next;
    setPersisted(next);
    savePersistence(next);
    void startMode(state.mode, value);
  };

  useEffect(() => {
    const snapshot = state.snapshot;
    if (!snapshot?.completed || recordedSessionRef.current === snapshot.sessionId) return;
    recordedSessionRef.current = snapshot.sessionId;
    const completed = persistedRef.current.completed.includes(lessonId)
      ? persistedRef.current.completed
      : [...persistedRef.current.completed, lessonId];
    const next = {
      ...persistedRef.current,
      completed,
      attempts: {
        ...persistedRef.current.attempts,
        [lessonId]:
          snapshot.mode === "challenge"
            ? Math.min(3, persistedRef.current.attempts[lessonId] + 1)
            : persistedRef.current.attempts[lessonId],
      },
    };
    persistedRef.current = next;
    setPersisted(next);
    savePersistence(next);
  }, [lessonId, state.snapshot]);

  const toggleInspector = () => {
    const open = !inspectorOpen;
    setInspectorOpen(open);
    const next = {
      ...persistedRef.current,
      preferences: { ...persistedRef.current.preferences, inspectorOpen: open },
    };
    persistedRef.current = next;
    setPersisted(next);
    savePersistence(next);
  };

  if (state.phase === "loading")
    return (
      <main className="lesson-page">
        <MissionHeader {...content} />
        <LoadingStages progress={progress} />
      </main>
    );
  if (state.phase === "unsupported")
    return (
      <main className="lesson-page">
        <MissionHeader {...content} />
        <UnsupportedBrowser reason={state.error ?? "Unsupported browser"} />
      </main>
    );
  if (state.phase === "error")
    return (
      <main className="lesson-page">
        <MissionHeader {...content} />
        <ErrorRecovery
          message={state.error ?? "Unknown runtime failure"}
          onRetry={() => {
            client.restart();
            dispatch({ type: "loading" });
            void client.initialize().then(() => startMode("guided"));
          }}
        />
      </main>
    );

  const explanation =
    explanationCopy[state.snapshot?.explanationKey ?? "ready"] ??
    "PyMAB updated its policy from the observed outcome.";
  return (
    <main className="lesson-page">
      <MissionHeader {...content} />
      <ModeTabs mode={state.mode} onChange={changeMode} />
      {(state.mode === "challenge" || state.mode === "freePlay") && (
        <ParameterChallenge
          label={content.parameterLabel}
          choices={content.choices}
          value={parameter}
          disabled={state.pending || autoRunning}
          target={content.target}
          onChange={changeParameter}
        />
      )}
      {state.mode === "freePlay" && (
        <label className="seed-input">
          Seed{" "}
          <input
            type="number"
            value={seed}
            disabled={state.pending || autoRunning}
            onChange={(event) => {
              const value = Number(event.target.value);
              setSeed(value);
              seedRef.current = value;
            }}
            onBlur={() => void startMode("freePlay", parameter, seed)}
          />
        </label>
      )}
      <div className="lesson-layout">
        <div className="game-column">
          <ProgressTrail snapshot={state.snapshot} />
          <Chamber
            snapshot={state.snapshot}
            animationState={
              state.pending ? "deciding" : state.snapshot?.reward === 1 ? "reward" : "idle"
            }
          />
          <OutcomeReveal snapshot={state.snapshot} explanation={explanation} />
          {!state.snapshot?.completed && (
            <RunControls
              pending={state.pending}
              completed={Boolean(state.snapshot?.completed)}
              autoRunning={autoRunning}
              onStep={() => void advance()}
              onAutoRun={autoRun}
              onPause={() => {
                autoRef.current = false;
                setAutoRunning(false);
              }}
              onReset={() => void startMode(state.mode)}
            />
          )}
          {state.snapshot?.completed && (
            <Debrief
              snapshot={state.snapshot}
              onChallenge={() => void startMode("challenge")}
              onFreePlay={() => void startMode("freePlay")}
            />
          )}
        </div>
        <InspectPanel
          snapshot={state.snapshot}
          open={inspectorOpen}
          onToggle={toggleInspector}
          onOpenLab={() => {
            void navigate("/lab", { state: { code: state.snapshot?.generatedCode, lessonId } });
          }}
        />
      </div>
    </main>
  );
}
