import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";

import {
  Chamber,
  Debrief,
  ErrorRecovery,
  InspectPanel,
  LoadingStages,
  MissionHeader,
  OutcomeReveal,
  ProgressTrail,
  RunControls,
  UnsupportedBrowser,
} from "../components/game";
import { RunSetupPanel } from "../components/game/RunSetupPanel";
import { explanationCopy, lessonContent } from "../content/lessons";
import type {
  LessonId,
  LessonMode,
  LessonRequest,
  LessonResponse,
  LessonSnapshot,
} from "../engine/protocol";
import { useRuntime } from "../engine/RuntimeProvider";
import { detectBrowserSupport } from "../engine/support";
import { WorkerClient } from "../engine/WorkerClient";
import { initialLessonState, lessonReducer } from "../state/lessonReducer";
import { loadPersistence, savePersistence } from "../state/persistence";
import {
  configurationFromSnapshot,
  draftFromConfiguration,
  parameterDefinitions,
  validateRunDraft,
  type RunConfiguration,
  type RunDraftConfiguration,
} from "../state/runConfiguration";
import { generatePortalProbabilities, probabilityDraft } from "../state/portalProbabilities";

interface LessonNavigationState {
  runConfiguration?: RunConfiguration;
}

function snapshotFrom(response: LessonResponse): LessonSnapshot {
  if ("snapshot" in response) return response.snapshot;
  if (response.type === "error") throw new Error(response.error.message);
  throw new Error(`Unexpected worker response: ${response.type}`);
}

function fixedSeed(lessonId: LessonId, mode: LessonMode): number {
  const content = lessonContent[lessonId];
  return mode === "challenge" ? content.challengeSeed : content.guidedSeed;
}

function defaultConfiguration(lessonId: LessonId): RunConfiguration {
  return {
    lessonId,
    mode: "guided",
    parameter: parameterDefinitions[lessonId].defaultValue,
    seed: lessonContent[lessonId].guidedSeed,
    portalProbabilities: null,
    probabilitySource: "generated",
  };
}

function configurationFromNavigationState(
  state: unknown,
  lessonId: LessonId,
): RunConfiguration | null {
  const candidate = (state as LessonNavigationState | null)?.runConfiguration;
  if (!candidate || candidate.lessonId !== lessonId) return null;
  return validateRunDraft(draftFromConfiguration(candidate)).configuration;
}

export function LessonRoute() {
  const { lessonSlug } = useParams();
  const lessonId: LessonId = lessonSlug === "linucb" ? "linucb" : "epsilon-greedy";
  const content = lessonContent[lessonId];
  const { client, progress } = useRuntime();
  const location = useLocation();
  const routeState = location.state as unknown;
  const navigate = useNavigate();
  const [state, dispatch] = useReducer(lessonReducer, initialLessonState);
  const [persisted, setPersisted] = useState(() => loadPersistence());
  const [activeConfiguration, setActiveConfiguration] = useState<RunConfiguration | null>(null);
  const [draftConfiguration, setDraftConfiguration] = useState<RunDraftConfiguration>(() =>
    draftFromConfiguration(defaultConfiguration(lessonId)),
  );
  const [inspectorOpen, setInspectorOpen] = useState(persisted.preferences.inspectorOpen);
  const [autoRunning, setAutoRunning] = useState(false);
  const autoRef = useRef(false);
  const sessionRef = useRef("");
  const persistedRef = useRef(persisted);
  const recordedSessionRef = useRef("");
  const locationStateRef = useRef<unknown>(routeState);

  useEffect(() => {
    locationStateRef.current = routeState;
  }, [routeState]);

  const startConfiguration = useCallback(
    async (configuration: RunConfiguration): Promise<boolean> => {
      autoRef.current = false;
      setAutoRunning(false);
      dispatch({ type: "pending", value: true });
      try {
        const previousSessionId = sessionRef.current;
        const sessionId = WorkerClient.requestId();
        const request: LessonRequest = {
          type: "startLesson",
          requestId: WorkerClient.requestId(),
          sessionId,
          lessonId: configuration.lessonId,
          mode: configuration.mode,
          seed: configuration.seed,
          parameters:
            configuration.lessonId === "epsilon-greedy"
              ? { epsilon: configuration.parameter }
              : { alpha: configuration.parameter, l2: 1 },
          ...(configuration.lessonId === "epsilon-greedy" &&
          configuration.mode === "freePlay" &&
          configuration.portalProbabilities
            ? { environment: { probabilities: configuration.portalProbabilities } }
            : {}),
        };
        const response = await client.send(request);
        const snapshot = snapshotFrom(response);
        const active = configurationFromSnapshot(snapshot, configuration.probabilitySource);
        sessionRef.current = sessionId;
        if (previousSessionId) {
          void client
            .send({
              type: "dispose",
              requestId: WorkerClient.requestId(),
              sessionId: previousSessionId,
            })
            .catch(() => undefined);
        }
        setActiveConfiguration(active);
        setDraftConfiguration(draftFromConfiguration(active));

        const previousRecent = persistedRef.current.recent[active.lessonId];
        const next = {
          ...persistedRef.current,
          recent: {
            ...persistedRef.current.recent,
            [active.lessonId]: {
              parameter: active.parameter,
              seed: active.mode === "freePlay" ? active.seed : previousRecent.seed,
            },
          },
        };
        persistedRef.current = next;
        setPersisted(next);
        savePersistence(next);
        dispatch({ type: "started", mode: active.mode, snapshot });
        return true;
      } catch (error) {
        dispatch({
          type: "error",
          message: error instanceof Error ? error.message : String(error),
        });
        return false;
      }
    },
    [client],
  );

  useEffect(() => {
    const support = detectBrowserSupport();
    if (!support.supported) {
      dispatch({
        type: "unsupported",
        message: support.reason ?? "Required browser features are unavailable.",
      });
      return;
    }

    const requestedConfiguration = configurationFromNavigationState(
      locationStateRef.current,
      lessonId,
    );
    const initialConfiguration = requestedConfiguration ?? defaultConfiguration(lessonId);

    let active = true;
    void client
      .initialize()
      .then(async () => {
        if (!active) return;
        const started = await startConfiguration(initialConfiguration);
        if (active && started && requestedConfiguration) {
          void navigate(`/lesson/${lessonId}`, { replace: true, state: null });
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
  }, [client, lessonId, navigate, startConfiguration]);

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

  const changeAlgorithm = (nextLessonId: LessonId) => {
    setDraftConfiguration((current) => ({
      ...current,
      lessonId: nextLessonId,
      parameter: String(parameterDefinitions[nextLessonId].defaultValue),
      seed:
        current.mode === "freePlay" ? current.seed : String(fixedSeed(nextLessonId, current.mode)),
    }));
  };

  const changeMode = (mode: LessonMode) => {
    setDraftConfiguration((current) => {
      const seed =
        mode === "freePlay"
          ? persistedRef.current.recent[current.lessonId].seed
          : fixedSeed(current.lessonId, mode);
      return {
        ...current,
        mode,
        seed: String(seed),
        ...(current.probabilitySource === "generated"
          ? { portalProbabilities: probabilityDraft(generatePortalProbabilities(seed)) }
          : {}),
      };
    });
  };

  const applyDraft = () => {
    const { configuration } = validateRunDraft(draftConfiguration);
    if (!configuration) return;
    if (configuration.lessonId !== lessonId) {
      autoRef.current = false;
      setAutoRunning(false);
      setActiveConfiguration(null);
      dispatch({ type: "loading" });
      void navigate(`/lesson/${configuration.lessonId}`, {
        state: { runConfiguration: configuration } satisfies LessonNavigationState,
      });
      return;
    }
    void startConfiguration(configuration);
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
            ? persistedRef.current.attempts[lessonId] + 1
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

  const setupPanel = (
    <RunSetupPanel
      activeConfiguration={activeConfiguration}
      draftConfiguration={draftConfiguration}
      challengeTarget={lessonContent[draftConfiguration.lessonId].target}
      pending={state.pending || autoRunning}
      onAlgorithmChange={changeAlgorithm}
      onModeChange={changeMode}
      onParameterChange={(parameter) =>
        setDraftConfiguration((current) => ({ ...current, parameter }))
      }
      onSeedChange={(seed) =>
        setDraftConfiguration((current) => {
          const numericSeed = Number(seed);
          return {
            ...current,
            seed,
            ...(current.probabilitySource === "generated" && Number.isSafeInteger(numericSeed)
              ? {
                  portalProbabilities: probabilityDraft(generatePortalProbabilities(numericSeed)),
                }
              : {}),
          };
        })
      }
      onPortalProbabilityChange={(index, value) =>
        setDraftConfiguration((current) => {
          const portalProbabilities = [...current.portalProbabilities] as [string, string, string];
          portalProbabilities[index] = value;
          return { ...current, portalProbabilities, probabilitySource: "custom" };
        })
      }
      onUseSeedGeneratedValues={() =>
        setDraftConfiguration((current) => {
          const seed = Number(current.seed);
          if (!Number.isSafeInteger(seed)) return current;
          return {
            ...current,
            portalProbabilities: probabilityDraft(generatePortalProbabilities(seed)),
            probabilitySource: "generated",
          };
        })
      }
      onApply={applyDraft}
    />
  );

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
        {setupPanel}
        <ErrorRecovery
          message={state.error ?? "Unknown runtime failure"}
          onRetry={() => {
            const retryConfiguration =
              validateRunDraft(draftConfiguration).configuration ?? defaultConfiguration(lessonId);
            client.restart();
            dispatch({ type: "loading" });
            void client.initialize().then(() => startConfiguration(retryConfiguration));
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
      {setupPanel}
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
              onReset={() => {
                if (activeConfiguration) void startConfiguration(activeConfiguration);
              }}
            />
          )}
          {state.snapshot?.completed && activeConfiguration && (
            <Debrief
              snapshot={state.snapshot}
              onChallenge={() =>
                void startConfiguration({
                  ...activeConfiguration,
                  mode: "challenge",
                  seed: content.challengeSeed,
                  portalProbabilities: null,
                  probabilitySource: "generated",
                })
              }
              onFreePlay={() => {
                const seed = persistedRef.current.recent[lessonId].seed;
                void startConfiguration({
                  ...activeConfiguration,
                  mode: "freePlay",
                  seed,
                  portalProbabilities:
                    lessonId === "epsilon-greedy" ? generatePortalProbabilities(seed) : null,
                  probabilitySource: "generated",
                });
              }}
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
