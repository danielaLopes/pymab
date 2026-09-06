import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { isScenarioId, scenarioCatalog, scenarioSeed } from "@/catalog/scenarios";
import type {
  LessonMode,
  LessonRequest,
  LessonResponse,
  LessonSnapshot,
  ScenarioId,
} from "@/engine/protocol";
import { WorkerClient } from "@/engine/WorkerClient";
import { initialLessonState, lessonReducer } from "@/state/lessonReducer";
import {
  cloneRecommendationCandidates,
  defaultRecommendationCandidates,
} from "@/state/scenarioCandidates";
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
import {
  ScenarioSetupPanel,
  type ScenarioConfiguration,
} from "../components/game/ScenarioSetupPanel";
import { useRuntime } from "../engine/RuntimeProvider";
import { detectBrowserSupport } from "../engine/support";

const scenarioExplanations: Record<string, string> = {
  "recommendations.1":
    "The policy began with equal click estimates, then observed feedback for only the item it showed.",
  "recommendations.2":
    "The same item can receive a different score when the visitor context changes.",
  "recommendations.3":
    "Only the selected item's logistic model is updated from this click or no-click result.",
  "recommendations.4":
    "Exploration supplies feedback for items that the current estimates might otherwise overlook.",
  "defensive-verification.1":
    "LinUCB combines its utility estimate with an uncertainty bonus for each approved action.",
  "defensive-verification.2":
    "Risk, account age and endpoint sensitivity change the action scores for this request.",
  "defensive-verification.3":
    "The observed utility rewards protection and low friction, while penalizing missed abuse and abandonment.",
  "defensive-verification.4":
    "The bandit learns only inside this review band. Deterministic allow and block rules remain outside it.",
  ready: "The policy is ready for the first decision.",
};

function environmentFor(scenarioId: ScenarioId): Record<string, unknown> {
  return scenarioId === "defensive-verification"
    ? { baseRisk: 0.32, riskWeight: 0.16, newAccountWeight: 0.07, sensitiveWeight: 0.06 }
    : {};
}

function environmentFromScenarioConfiguration(
  configuration: ScenarioConfiguration,
): Record<string, unknown> | null {
  if (configuration.scenarioId !== "recommendations") return configuration.environment;
  const candidates = configuration.candidates ?? cloneRecommendationCandidates();
  return {
    candidates: candidates.map(({ id, name, symbolKind }) => ({
      id,
      name: name.trim(),
      symbolKind,
    })),
    theta: candidates.map((candidate) => [...candidate.coefficients]),
  };
}

function defaultScenarioConfiguration(
  scenarioId: ScenarioId,
  mode: LessonMode = "guided",
): ScenarioConfiguration {
  return {
    scenarioId,
    mode,
    seed: scenarioSeed(scenarioId, mode),
    parameters: { ...scenarioCatalog[scenarioId].parameters },
    environment:
      scenarioId === "defensive-verification" && mode === "freePlay"
        ? environmentFor(scenarioId)
        : null,
    candidates:
      scenarioId === "recommendations"
        ? cloneRecommendationCandidates(defaultRecommendationCandidates)
        : null,
    nextCandidateOrdinal: 4,
  };
}

function snapshotFrom(response: LessonResponse): LessonSnapshot {
  if ("snapshot" in response) return response.snapshot;
  if (response.type === "error") throw new Error(response.error.message);
  throw new Error(`Unexpected worker response: ${response.type}`);
}

export function ScenarioRoute() {
  const { scenarioSlug } = useParams();
  const scenarioId: ScenarioId =
    scenarioSlug && isScenarioId(scenarioSlug) ? scenarioSlug : "recommendations";
  return <ScenarioExperience scenarioId={scenarioId} />;
}

function ScenarioExperience({ scenarioId }: { scenarioId: ScenarioId }) {
  const definition = scenarioCatalog[scenarioId];
  const { client, progress } = useRuntime();
  const navigate = useNavigate();
  const [state, dispatch] = useReducer(lessonReducer, initialLessonState);
  const [configuration, setConfiguration] = useState(() =>
    defaultScenarioConfiguration(scenarioId),
  );
  const [activeConfiguration, setActiveConfiguration] = useState<ScenarioConfiguration | null>(
    null,
  );
  const [autoRunning, setAutoRunning] = useState(false);
  const [setupOpen, setSetupOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const autoRef = useRef(false);
  const sessionRef = useRef("");

  const start = useCallback(
    async (next: ScenarioConfiguration): Promise<LessonSnapshot | null> => {
      autoRef.current = false;
      setAutoRunning(false);
      dispatch({ type: "pending", value: true });
      try {
        const previous = sessionRef.current;
        if (previous) {
          sessionRef.current = "";
          await client.send({
            type: "dispose",
            requestId: WorkerClient.requestId(),
            sessionId: previous,
          });
        }
        const sessionId = WorkerClient.requestId();
        const environment = environmentFromScenarioConfiguration(next);
        const request: LessonRequest = {
          type: "startScenario",
          requestId: WorkerClient.requestId(),
          sessionId,
          scenarioId: next.scenarioId,
          mode: next.mode,
          seed: next.seed,
          parameters: next.parameters,
          ...(environment ? { environment } : {}),
        };
        const snapshot = snapshotFrom(await client.send(request));
        sessionRef.current = sessionId;
        setActiveConfiguration(next);
        setSetupOpen(false);
        dispatch({ type: "started", mode: next.mode, snapshot });
        return snapshot;
      } catch (error) {
        dispatch({
          type: "error",
          message: error instanceof Error ? error.message : String(error),
        });
        return null;
      }
    },
    [client],
  );

  useEffect(() => {
    const next = defaultScenarioConfiguration(scenarioId);
    let active = true;
    void (async () => {
      await Promise.resolve();
      if (!active) return;
      setConfiguration(next);
      setActiveConfiguration(null);
      dispatch({ type: "loading" });
      const support = detectBrowserSupport();
      if (!support.supported) {
        dispatch({
          type: "unsupported",
          message: support.reason ?? "Required browser features are unavailable.",
        });
        return;
      }
      try {
        await client.initialize();
        if (active) await start(next);
      } catch (error: unknown) {
        if (active) {
          dispatch({
            type: "error",
            message: error instanceof Error ? error.message : String(error),
          });
        }
      }
    })();
    return () => {
      active = false;
      autoRef.current = false;
    };
  }, [client, scenarioId, start]);

  useEffect(
    () => () => {
      const sessionId = sessionRef.current;
      sessionRef.current = "";
      if (sessionId)
        void client
          .send({ type: "dispose", requestId: WorkerClient.requestId(), sessionId })
          .catch(() => undefined);
    },
    [client],
  );

  const advance = useCallback(async (): Promise<LessonSnapshot | null> => {
    if (!sessionRef.current) return null;
    dispatch({ type: "pending", value: true });
    try {
      const snapshot = snapshotFrom(
        await client.send({
          type: "step",
          requestId: WorkerClient.requestId(),
          sessionId: sessionRef.current,
        }),
      );
      dispatch({ type: "snapshot", snapshot });
      return snapshot;
    } catch (error) {
      dispatch({ type: "error", message: error instanceof Error ? error.message : String(error) });
      return null;
    }
  }, [client]);

  const autoRun = () => {
    autoRef.current = true;
    setAutoRunning(true);
    void (async () => {
      while (autoRef.current) {
        const snapshot = await advance();
        if (!snapshot || snapshot.completed) break;
        await new Promise((resolve) => window.setTimeout(resolve, 360));
      }
      setAutoRunning(false);
    })();
  };

  const header = (
    <MissionHeader eyebrow={definition.eyebrow} title={definition.title} intro={definition.intro} />
  );
  const setup = (
    <ScenarioSetupPanel
      configuration={configuration}
      pending={state.pending || autoRunning}
      expanded={setupOpen}
      onChange={(next) =>
        setConfiguration({
          ...next,
          environment:
            next.scenarioId === "defensive-verification" && next.mode === "freePlay"
              ? (next.environment ?? environmentFor(next.scenarioId))
              : next.environment,
        })
      }
      onScenarioChange={(next) => void navigate(`/scenario/${next}`)}
      onExpandedChange={setSetupOpen}
      onApply={() => void start(configuration)}
    />
  );

  if (state.phase === "loading")
    return (
      <main className="lesson-page">
        {header}
        <LoadingStages progress={progress} />
      </main>
    );
  if (state.phase === "unsupported")
    return (
      <main className="lesson-page">
        {header}
        <UnsupportedBrowser reason={state.error ?? "Unsupported browser"} />
      </main>
    );
  if (state.phase === "error")
    return (
      <main className="lesson-page">
        {header}
        {setup}
        <ErrorRecovery
          message={state.error ?? "Unknown runtime failure"}
          onRetry={() => {
            client.restart();
            dispatch({ type: "loading" });
            void client.initialize().then(() => start(configuration));
          }}
        />
      </main>
    );

  const explanation =
    scenarioExplanations[state.snapshot?.explanationKey ?? "ready"] ??
    "PyMAB updated the selected action from the observed result.";
  return (
    <main className="lesson-page">
      {header}
      <section className="scenario-boundary" aria-label="Scenario boundary">
        <div>
          <strong>{scenarioId === "recommendations" ? "Good fit" : "Bandit scope"}</strong>
          <p>
            {scenarioId === "recommendations"
              ? "Choose one item and learn from an immediate click or no-click result."
              : "Choose among approved, reversible checks for requests in the review band."}
          </p>
        </div>
        <div>
          <strong>{scenarioId === "recommendations" ? "Not covered" : "Fixed rules"}</strong>
          <p>
            {scenarioId === "recommendations"
              ? "Ranking, delayed purchases and long-term satisfaction need different methods."
              : "Clear allows and hard blocks stay outside the policy, as does the upstream risk model."}
          </p>
        </div>
      </section>
      {setup}
      <div className="scenario-run-stack">
        <div className="game-column">
          <ProgressTrail snapshot={state.snapshot} />
          <DecisionHistoryBoard
            snapshot={state.snapshot}
            pending={state.pending}
            eyebrow="Decision history"
            title="Context, scores and observed outcomes by round"
            spacious
          />
          <OutcomeReveal snapshot={state.snapshot} explanation={explanation} />
          {!state.snapshot?.completed && (
            <RunControls
              pending={state.pending}
              completed={false}
              autoRunning={autoRunning}
              onStep={() => void advance()}
              onAutoRun={autoRun}
              onPause={() => {
                autoRef.current = false;
                setAutoRunning(false);
              }}
              onReset={() => {
                if (activeConfiguration) void start(activeConfiguration);
              }}
            />
          )}
          {state.snapshot?.completed && activeConfiguration && (
            <Debrief
              snapshot={state.snapshot}
              onChallenge={() => {
                const next = defaultScenarioConfiguration(scenarioId, "challenge");
                setConfiguration(next);
                void start(next);
              }}
              onFreePlay={() => {
                const next = defaultScenarioConfiguration(scenarioId, "freePlay");
                setConfiguration(next);
                void start(next);
              }}
            />
          )}
        </div>
        <div className="scenario-developer-view">
          <InspectPanel
            snapshot={state.snapshot}
            open={inspectorOpen}
            onToggle={() => setInspectorOpen((open) => !open)}
            onOpenLab={() =>
              void navigate("/lab", {
                state: { code: state.snapshot?.generatedCode, lessonId: definition.policyId },
              })
            }
          />
        </div>
      </div>
    </main>
  );
}
