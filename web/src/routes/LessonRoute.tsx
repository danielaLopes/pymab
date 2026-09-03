import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";

import { isPolicyId, policyCatalog, type PolicyId } from "@/catalog/policies";

import {
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
import { DecisionHistoryBoard } from "../components/game/DecisionHistoryBoard";
import { RunSetupPanel } from "../components/game/RunSetupPanel";
import { explanationCopy, lessonContent } from "../content/lessons";
import type { LessonMode, LessonRequest, LessonResponse, LessonSnapshot } from "../engine/protocol";
import { useRuntime } from "../engine/RuntimeProvider";
import { detectBrowserSupport } from "../engine/support";
import { WorkerClient } from "../engine/WorkerClient";
import { initialLessonState, lessonReducer } from "../state/lessonReducer";
import { loadPersistence, savePersistence } from "../state/persistence";
import {
  configurationFromSnapshot,
  defaultConfiguration,
  draftFromConfiguration,
  regenerateDraftEnvironment,
  validateRunDraft,
  type RunConfiguration,
  type RunDraftConfiguration,
} from "../state/runConfiguration";

interface LessonNavigationState {
  runConfiguration?: RunConfiguration;
}

function snapshotFrom(response: LessonResponse): LessonSnapshot {
  if ("snapshot" in response) return response.snapshot;
  if (response.type === "error") throw new Error(response.error.message);
  throw new Error(`Unexpected worker response: ${response.type}`);
}

function configurationFromNavigationState(
  state: unknown,
  policyId: PolicyId,
): RunConfiguration | null {
  const candidate = (state as LessonNavigationState | null)?.runConfiguration;
  if (!candidate || candidate.policyId !== policyId) return null;
  return validateRunDraft(draftFromConfiguration(candidate)).configuration;
}

export function LessonRoute() {
  const { lessonSlug } = useParams();
  const policyId: PolicyId = lessonSlug && isPolicyId(lessonSlug) ? lessonSlug : "epsilon-greedy";
  const content = lessonContent[policyId];
  const { client, progress } = useRuntime();
  const location = useLocation();
  const routeState = location.state as unknown;
  const navigate = useNavigate();
  const [state, dispatch] = useReducer(lessonReducer, initialLessonState);
  const [persisted, setPersisted] = useState(() => loadPersistence());
  const [activeConfiguration, setActiveConfiguration] = useState<RunConfiguration | null>(null);
  const [draftConfiguration, setDraftConfiguration] = useState<RunDraftConfiguration>(() =>
    draftFromConfiguration(defaultConfiguration(policyId)),
  );
  const [inspectorOpen, setInspectorOpen] = useState(persisted.preferences.inspectorOpen);
  const [autoRunning, setAutoRunning] = useState(false);
  const autoRef = useRef(false);
  const sessionRef = useRef("");
  const routeTransitionRef = useRef<Promise<void>>(Promise.resolve());
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
        if (previousSessionId) {
          sessionRef.current = "";
          await client.send({
            type: "dispose",
            requestId: WorkerClient.requestId(),
            sessionId: previousSessionId,
          });
        }
        const sessionId = WorkerClient.requestId();
        const request: LessonRequest = {
          type: "startLesson",
          requestId: WorkerClient.requestId(),
          sessionId,
          policyId: configuration.policyId,
          mode: configuration.mode,
          seed: configuration.seed,
          parameters: configuration.parameters,
          ...(configuration.environment ? { environment: configuration.environment } : {}),
        };
        const response = await client.send(request);
        const snapshot = snapshotFrom(response);
        const active = configurationFromSnapshot(snapshot, configuration.probabilitySource);
        sessionRef.current = sessionId;
        setActiveConfiguration(active);
        setDraftConfiguration(draftFromConfiguration(active));

        const previousRecent = persistedRef.current.recent[active.policyId];
        const next = {
          ...persistedRef.current,
          recent: {
            ...persistedRef.current.recent,
            [active.policyId]: {
              parameters: active.parameters,
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
      policyId,
    );
    const initialConfiguration = requestedConfiguration ?? defaultConfiguration(policyId);

    let active = true;
    const transition = routeTransitionRef.current
      .catch(() => undefined)
      .then(async () => {
        try {
          await client.initialize();
          if (!active) return;
          const started = await startConfiguration(initialConfiguration);
          if (active && started && requestedConfiguration) {
            void navigate(`/lesson/${policyId}`, { replace: true, state: null });
          }
        } catch (error: unknown) {
          if (active) {
            dispatch({
              type: "error",
              message: error instanceof Error ? error.message : String(error),
            });
          }
        }
      });
    routeTransitionRef.current = transition;
    return () => {
      active = false;
      autoRef.current = false;
    };
  }, [client, policyId, navigate, startConfiguration]);

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
        .catch(() => undefined);
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

  const changePolicy = (nextPolicyId: PolicyId) => {
    if (nextPolicyId === policyId) return;
    const next = defaultConfiguration(nextPolicyId, draftConfiguration.mode);
    if (draftConfiguration.mode === "freePlay") {
      next.seed = persistedRef.current.recent[nextPolicyId].seed;
      next.parameters = { ...persistedRef.current.recent[nextPolicyId].parameters };
    }
    autoRef.current = false;
    setAutoRunning(false);
    setActiveConfiguration(null);
    dispatch({ type: "loading" });
    void navigate(`/lesson/${nextPolicyId}`, {
      state: { runConfiguration: next } satisfies LessonNavigationState,
    });
  };

  const changeMode = (mode: LessonMode) => {
    setDraftConfiguration((current) => {
      const seed =
        mode === "freePlay"
          ? persistedRef.current.recent[current.policyId].seed
          : mode === "challenge"
            ? policyCatalog[current.policyId].challengeSeed
            : policyCatalog[current.policyId].guidedSeed;
      const parameters =
        mode === "freePlay"
          ? persistedRef.current.recent[current.policyId].parameters
          : policyCatalog[current.policyId].guidedParameters;
      return {
        ...current,
        mode,
        seed: String(seed),
        parameters: Object.fromEntries(
          Object.entries(parameters).map(([key, value]) => [
            key,
            typeof value === "boolean" ? value : value === null ? "" : String(value),
          ]),
        ),
        ...(current.probabilitySource === "generated"
          ? { environment: regenerateDraftEnvironment(current.policyId, seed) }
          : {}),
      };
    });
  };

  const applyDraft = () => {
    const { configuration } = validateRunDraft(draftConfiguration);
    if (!configuration) return;
    if (configuration.policyId !== policyId) {
      autoRef.current = false;
      setAutoRunning(false);
      setActiveConfiguration(null);
      dispatch({ type: "loading" });
      void navigate(`/lesson/${configuration.policyId}`, {
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
    const completed = persistedRef.current.completed.includes(policyId)
      ? persistedRef.current.completed
      : [...persistedRef.current.completed, policyId];
    const next = {
      ...persistedRef.current,
      completed,
      attempts: {
        ...persistedRef.current.attempts,
        [policyId]:
          snapshot.mode === "challenge"
            ? persistedRef.current.attempts[policyId] + 1
            : persistedRef.current.attempts[policyId],
      },
    };
    persistedRef.current = next;
    setPersisted(next);
    savePersistence(next);
  }, [policyId, state.snapshot]);

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
      challengeTarget={lessonContent[draftConfiguration.policyId].target}
      pending={state.pending || autoRunning}
      onPolicyChange={changePolicy}
      onModeChange={changeMode}
      onParameterChange={(key, value) =>
        setDraftConfiguration((current) => ({
          ...current,
          parameters: { ...current.parameters, [key]: value },
        }))
      }
      onSeedChange={(seed) =>
        setDraftConfiguration((current) => {
          const numericSeed = Number(seed);
          return {
            ...current,
            seed,
            ...(current.probabilitySource === "generated" && Number.isSafeInteger(numericSeed)
              ? { environment: regenerateDraftEnvironment(current.policyId, numericSeed) }
              : {}),
          };
        })
      }
      onEnvironmentChange={(environment) =>
        setDraftConfiguration((current) => ({
          ...current,
          environment,
          probabilitySource: "custom",
        }))
      }
      onRegenerateEnvironment={() =>
        setDraftConfiguration((current) => {
          const seed = Number(current.seed);
          if (!Number.isSafeInteger(seed)) return current;
          return {
            ...current,
            environment: regenerateDraftEnvironment(current.policyId, seed),
            probabilitySource: "generated",
          };
        })
      }
      onRestorePolicyDefaults={() =>
        setDraftConfiguration((current) => ({
          ...current,
          parameters: Object.fromEntries(
            Object.entries(policyCatalog[current.policyId].defaults).map(([key, value]) => [
              key,
              typeof value === "boolean" ? value : value === null ? "" : String(value),
            ]),
          ),
        }))
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
              validateRunDraft(draftConfiguration).configuration ?? defaultConfiguration(policyId);
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
          <DecisionHistoryBoard snapshot={state.snapshot} pending={state.pending} />
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
                  environment: null,
                  probabilitySource: "generated",
                })
              }
              onFreePlay={() => {
                const seed = persistedRef.current.recent[policyId].seed;
                void startConfiguration({
                  ...activeConfiguration,
                  mode: "freePlay",
                  seed,
                  environment:
                    validateRunDraft({
                      ...draftFromConfiguration({
                        ...activeConfiguration,
                        mode: "freePlay",
                        seed,
                        environment: null,
                        probabilitySource: "generated",
                      }),
                      mode: "freePlay",
                    }).configuration?.environment ?? null,
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
            void navigate("/lab", {
              state: { code: state.snapshot?.generatedCode, lessonId: policyId },
            });
          }}
        />
      </div>
    </main>
  );
}
