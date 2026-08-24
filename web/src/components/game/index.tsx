import { useEffect, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { Link } from "react-router-dom";

import type { LessonId, LessonSnapshot, RuntimeProgress } from "../../engine/protocol";
import { loadPersistence, savePersistence } from "../../state/persistence";

const gateDetails = [
  { name: "Moon Gate", symbol: "☾", rune: "Memory" },
  { name: "Sun Gate", symbol: "☼", rune: "Promise" },
  { name: "Star Gate", symbol: "✦", rune: "Possibility" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const [motionOverride, setMotionOverride] = useState<boolean | null>(
    () => loadPersistence().preferences.reducedMotion,
  );
  useEffect(() => {
    const systemReduced =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    document.documentElement.dataset.reducedMotion = String(motionOverride ?? systemReduced);
  }, [motionOverride]);
  const cycleMotion = () => {
    const next = motionOverride === null ? true : motionOverride ? false : null;
    setMotionOverride(next);
    const persisted = loadPersistence();
    savePersistence({
      ...persisted,
      preferences: { ...persisted.preferences, reducedMotion: next },
    });
  };
  return (
    <div className="app-shell">
      <header className="site-header">
        <Link className="brand" to="/" aria-label="PyMAB Arcade home">
          <span className="brand-mark" aria-hidden="true">
            P
          </span>
          <span>
            PyMAB <b>Arcade</b>
          </span>
        </Link>
        <nav aria-label="Primary navigation">
          <Link to="/">Missions</Link>
          <Link to="/lab">Python Lab</Link>
          <button className="motion-toggle" type="button" onClick={cycleMotion}>
            Motion: {motionOverride === null ? "system" : motionOverride ? "reduced" : "full"}
          </button>
          <a href="https://github.com/danielaLopes/pymab">GitHub</a>
        </nav>
      </header>
      {children}
      <footer>Runs locally in your browser · No account · No analytics</footer>
    </div>
  );
}

export function CampaignMap() {
  return (
    <div className="campaign-grid">
      <Link className="mission-card mission-one" to="/lesson/epsilon-greedy">
        <span className="mission-number">01</span>
        <span className="tag">FOUNDATIONS</span>
        <h2>The Three Ancient Gates</h2>
        <p>Discover why learning requires both curiosity and conviction.</p>
        <span className="mission-cta">
          Enter the chamber <span aria-hidden="true">→</span>
        </span>
      </Link>
      <Link className="mission-card mission-two" to="/lesson/linucb">
        <span className="mission-number">02</span>
        <span className="tag">CONTEXTUAL BANDITS</span>
        <h2>The Labyrinth of Signals</h2>
        <p>Read a changing world and act with calibrated confidence.</p>
        <span className="mission-cta">
          Follow the signals <span aria-hidden="true">→</span>
        </span>
      </Link>
    </div>
  );
}

export function MissionHeader({
  eyebrow,
  title,
  intro,
}: {
  eyebrow: string;
  title: string;
  intro: string;
}) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => heading.current?.focus(), [title]);
  return (
    <header className="mission-header">
      <Link to="/" className="back-link">
        ← Mission map
      </Link>
      <p className="eyebrow">{eyebrow}</p>
      <h1 ref={heading} tabIndex={-1}>
        {title}
      </h1>
      <p>{intro}</p>
    </header>
  );
}

export function CueStrip({ snapshot }: { snapshot: LessonSnapshot | null }) {
  if (!snapshot?.visibleCues.length)
    return <p className="cue-empty">No signals in this mission—the gates stay the same.</p>;
  return (
    <ul className="cue-strip" aria-label="Current chamber signals">
      {snapshot.visibleCues.map((cue) => (
        <li key={cue.name}>
          <span aria-hidden="true">
            {cue.name === "light" ? "◐" : cue.name === "echo" ? "≋" : "≈"}
          </span>
          <small>{cue.name}</small>
          <strong>{cue.label.replace(` ${cue.name}`, "")}</strong>
          <code>{cue.value > 0 ? "+1" : "−1"}</code>
        </li>
      ))}
    </ul>
  );
}

export function Gate({
  index,
  selected,
  reward,
  onInspect,
}: {
  index: number;
  selected: boolean;
  reward: number | null;
  onInspect: () => void;
}) {
  const gate = gateDetails[index]!;
  return (
    <button
      type="button"
      className={`gate gate-${index} ${selected ? "selected" : ""}`}
      aria-label={`${gate.name}, ${gate.rune}${selected ? ", selected by PyMAB" : ""}`}
      onClick={onInspect}
    >
      <span className="gate-arch" aria-hidden="true">
        <span className="gate-symbol">{gate.symbol}</span>
        {selected && <span className="gate-glow" />}
      </span>
      <strong>{gate.name}</strong>
      <small>{gate.rune}</small>
      {selected && reward !== null && (
        <span className={`reward-token ${reward ? "won" : "empty"}`}>
          {reward ? "+1 RELIC" : "EMPTY"}
        </span>
      )}
    </button>
  );
}

export function Chamber({
  snapshot,
  animationState = "idle",
}: {
  snapshot: LessonSnapshot | null;
  animationState?: string;
}) {
  return (
    <section className={`chamber ${animationState}`} aria-label="Infinite Crossroads chamber">
      <div className="chamber-haze" aria-hidden="true" />
      <CueStrip snapshot={snapshot} />
      <div className="gates">
        {gateDetails.map((_, index) => (
          <Gate
            key={index}
            index={index}
            selected={snapshot?.selectedArm === index}
            reward={snapshot?.reward ?? null}
            onInspect={() => undefined}
          />
        ))}
      </div>
    </section>
  );
}

export function OutcomeReveal({
  snapshot,
  explanation,
}: {
  snapshot: LessonSnapshot | null;
  explanation: string;
}) {
  return (
    <div className="outcome" aria-live="polite">
      <span className="outcome-icon" aria-hidden="true">
        {snapshot?.reward === 1 ? "✦" : "◇"}
      </span>
      <div>
        <strong>
          {snapshot?.step
            ? `Chamber ${snapshot.step}: ${snapshot.reward ? "Relic found" : "No relic this time"}`
            : "Awaiting the first decision"}
        </strong>
        <p>{explanation}</p>
      </div>
    </div>
  );
}

export function ProgressTrail({ snapshot }: { snapshot: LessonSnapshot | null }) {
  const horizon = snapshot?.horizon ?? 12;
  const step = snapshot?.step ?? 0;
  return (
    <div className="progress-trail">
      <div>
        <span>Expedition</span>
        <strong>
          {step} / {horizon}
        </strong>
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-label="Expedition progress"
        aria-valuemin={0}
        aria-valuemax={horizon}
        aria-valuenow={step}
      >
        <span style={{ width: `${(step / horizon) * 100}%` }} />
      </div>
      <dl>
        <div>
          <dt>Relics</dt>
          <dd>{snapshot?.totalReward ?? 0}</dd>
        </div>
        <div>
          <dt>Expected regret</dt>
          <dd>{(snapshot?.cumulativeExpectedRegret ?? 0).toFixed(2)}</dd>
        </div>
      </dl>
    </div>
  );
}

export function RunControls({
  pending,
  completed,
  autoRunning,
  onStep,
  onAutoRun,
  onPause,
  onReset,
}: {
  pending: boolean;
  completed: boolean;
  autoRunning: boolean;
  onStep: () => void;
  onAutoRun: () => void;
  onPause: () => void;
  onReset: () => void;
}) {
  return (
    <div className="run-controls">
      <button
        className="primary-button"
        type="button"
        disabled={pending || completed || autoRunning}
        onClick={onStep}
      >
        Advance one chamber
      </button>
      {autoRunning ? (
        <button type="button" onClick={onPause}>
          Pause expedition
        </button>
      ) : (
        <button type="button" disabled={pending || completed} onClick={onAutoRun}>
          Auto-run
        </button>
      )}
      <button type="button" disabled={pending} onClick={onReset}>
        Reset seed
      </button>
    </div>
  );
}

export function ParameterChallenge({
  label,
  choices,
  value,
  disabled,
  target,
  onChange,
}: {
  label: string;
  choices: number[];
  value: number;
  disabled: boolean;
  target: string;
  onChange: (value: number) => void;
}) {
  return (
    <fieldset className="parameter-challenge" disabled={disabled}>
      <legend>{label}</legend>
      <div className="choice-row">
        {choices.map((choice) => (
          <label key={choice} className={choice === value ? "active" : ""}>
            <input
              type="radio"
              name="parameter"
              value={choice}
              checked={choice === value}
              onChange={() => onChange(choice)}
            />
            {choice}
          </label>
        ))}
      </div>
      <p>{target}</p>
    </fieldset>
  );
}

export function Debrief({
  snapshot,
  onChallenge,
  onFreePlay,
}: {
  snapshot: LessonSnapshot;
  onChallenge: () => void;
  onFreePlay: () => void;
}) {
  return (
    <section className={`debrief ${snapshot.passed ? "passed" : "complete"}`}>
      <p className="eyebrow">Expedition complete</p>
      <h2>
        {snapshot.mode === "challenge"
          ? snapshot.passed
            ? "Challenge cleared"
            : "A useful failure"
          : "The map has learned from you"}
      </h2>
      <p>
        You collected <strong>{snapshot.totalReward} relics</strong> with{" "}
        <strong>{snapshot.cumulativeExpectedRegret.toFixed(2)} expected regret</strong>.
      </p>
      <p className="caveat">
        One seeded expedition illustrates behaviour; it does not prove that a parameter is
        universally best.
      </p>
      <div className="run-controls">
        <button className="primary-button" onClick={onChallenge}>
          Try the challenge
        </button>
        <button onClick={onFreePlay}>Enter free play</button>
      </div>
    </section>
  );
}

export function LoadingStages({ progress }: { progress: RuntimeProgress | null }) {
  return (
    <div className="loading-panel" role="status">
      <span className="loader" aria-hidden="true" />
      <div>
        <strong>Waking the Python engine</strong>
        <p>{progress?.message ?? "Preparing the chamber…"}</p>
      </div>
    </div>
  );
}

export function ErrorRecovery({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="error-panel" role="alert">
      <h2>The chamber lost its signal</h2>
      <p>{message}</p>
      <button onClick={onRetry}>Retry from step zero</button>
    </div>
  );
}

export function UnsupportedBrowser({ reason }: { reason: string }) {
  return (
    <div className="error-panel">
      <h2>This browser cannot open the Arcade</h2>
      <p>{reason}</p>
      <p>You can still read about each policy in the PyMAB documentation.</p>
    </div>
  );
}

export function PolicyBars({ snapshot }: { snapshot: LessonSnapshot }) {
  const diagnostic = snapshot.diagnostic;
  if (!diagnostic) return null;
  const values =
    snapshot.lessonId === "epsilon-greedy" ? diagnostic.estimatesAfter : diagnostic.ucbScores;
  if (!Array.isArray(values)) return null;
  const numeric = values.map(Number);
  const scale = Math.max(...numeric.map(Math.abs), 1);
  if (snapshot.lessonId === "linucb") {
    const means = Array.isArray(diagnostic.predictedMeans)
      ? diagnostic.predictedMeans.map(Number)
      : [];
    const bonuses = Array.isArray(diagnostic.bonuses) ? diagnostic.bonuses.map(Number) : [];
    return (
      <table className="score-table">
        <caption>LinUCB score decomposition</caption>
        <thead>
          <tr>
            <th>Gate</th>
            <th>Prediction</th>
            <th>Bonus</th>
            <th>UCB</th>
          </tr>
        </thead>
        <tbody>
          {numeric.map((value, index) => (
            <tr key={index}>
              <th>{gateDetails[index]?.name}</th>
              <td>{means[index]?.toPrecision(4)}</td>
              <td>+ {bonuses[index]?.toPrecision(4)}</td>
              <td>
                <strong>{value.toPrecision(4)}</strong>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  return (
    <div
      className="policy-bars"
      role="img"
      aria-label={`${snapshot.lessonId} scores: ${numeric.map((value, i) => `${gateDetails[i]?.name} ${value.toPrecision(4)}`).join(", ")}`}
    >
      {numeric.map((value, index) => (
        <div key={index}>
          <span>{gateDetails[index]?.symbol}</span>
          <i
            style={{ "--bar": `${Math.max(3, (Math.abs(value) / scale) * 100)}%` } as CSSProperties}
          />
          <code>{value.toPrecision(4)}</code>
        </div>
      ))}
    </div>
  );
}

function MatrixTable({ label, value }: { label: string; value: unknown }) {
  if (!Array.isArray(value) || !value.every((row) => Array.isArray(row))) return null;
  const matrix = value as unknown[][];
  return (
    <table className="matrix-table">
      <caption>{label}</caption>
      <tbody>
        {matrix.map((row, rowIndex) => (
          <tr key={rowIndex}>
            <th scope="row">Gate {rowIndex + 1}</th>
            {row.map((cell, columnIndex) => (
              <td key={columnIndex}>{Number(cell).toPrecision(4)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function InspectPanel({
  snapshot,
  open,
  onToggle,
  onOpenLab,
}: {
  snapshot: LessonSnapshot | null;
  open: boolean;
  onToggle: () => void;
  onOpenLab: () => void;
}) {
  const [copyStatus, setCopyStatus] = useState("");
  const copyCode = () => {
    if (!snapshot) return;
    void navigator.clipboard.writeText(snapshot.generatedCode).then(
      () => setCopyStatus("Python copied."),
      () => setCopyStatus("Copy failed. Select the code manually."),
    );
  };
  return (
    <aside className="inspector">
      <button className="inspector-toggle" type="button" aria-expanded={open} onClick={onToggle}>
        <span>
          <small>Developer view</small>
          <strong>Inspect PyMAB</strong>
        </span>
        <span aria-hidden="true">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="inspector-body">
          {!snapshot ? (
            <p>Run a chamber to inspect the policy state.</p>
          ) : (
            <>
              <dl className="metadata">
                <div>
                  <dt>Class</dt>
                  <dd>
                    {snapshot.lessonId === "epsilon-greedy"
                      ? "EpsilonGreedyPolicy"
                      : "LinUCBPolicy"}
                  </dd>
                </div>
                <div>
                  <dt>PyMAB</dt>
                  <dd>{snapshot.packageVersion}</dd>
                </div>
                <div>
                  <dt>Seed</dt>
                  <dd>{snapshot.seed}</dd>
                </div>
                <div>
                  <dt>Constructor</dt>
                  <dd>
                    <code>
                      {snapshot.lessonId === "epsilon-greedy"
                        ? `EpsilonGreedyPolicy(n_arms=3, epsilon=${snapshot.parameters.epsilon})`
                        : `LinUCBPolicy(n_arms=3, n_features=4, alpha=${snapshot.parameters.alpha}, l2=${snapshot.parameters.l2})`}
                    </code>
                  </dd>
                </div>
                <div>
                  <dt>Commit</dt>
                  <dd>
                    <code>{snapshot.sourceCommit.slice(0, 8)}</code>
                  </dd>
                </div>
              </dl>
              <h2>Decision state</h2>
              <PolicyBars snapshot={snapshot} />
              {snapshot.lessonId === "linucb" && snapshot.diagnostic && (
                <>
                  <MatrixTable
                    label="Current context matrix"
                    value={snapshot.diagnostic.contextMatrix}
                  />
                  <MatrixTable
                    label="Learned coefficient estimates"
                    value={snapshot.diagnostic.thetaBefore}
                  />
                </>
              )}
              <details>
                <summary>Full validated snapshot</summary>
                <pre>{JSON.stringify(snapshot.diagnostic, null, 2)}</pre>
              </details>
              {snapshot.hiddenTruth && (
                <details>
                  <summary>Revealed environment truth</summary>
                  <pre>{JSON.stringify(snapshot.hiddenTruth, null, 2)}</pre>
                </details>
              )}
              <h2>Equivalent Python</h2>
              <pre className="code-preview" tabIndex={0} aria-label="Equivalent Python code">
                <code>{snapshot.generatedCode}</code>
              </pre>
              <div className="run-controls">
                <button onClick={copyCode}>Copy code</button>
                <button className="primary-button" onClick={onOpenLab}>
                  Open in Python Lab
                </button>
              </div>
              <p className="copy-status" aria-live="polite">
                {copyStatus}
              </p>
            </>
          )}
        </div>
      )}
    </aside>
  );
}

export function ModeTabs({
  mode,
  onChange,
}: {
  mode: string;
  onChange: (mode: "guided" | "challenge" | "freePlay") => void;
}) {
  return (
    <div className="mode-tabs" aria-label="Lesson mode">
      {(["guided", "challenge", "freePlay"] as const).map((item) => (
        <button
          key={item}
          type="button"
          aria-pressed={mode === item}
          onClick={() => onChange(item)}
        >
          {item === "freePlay" ? "Free play" : item[0]!.toUpperCase() + item.slice(1)}
        </button>
      ))}
    </div>
  );
}

export function LessonBadge({ lessonId }: { lessonId: LessonId }) {
  return <span className="lesson-badge">{lessonId === "epsilon-greedy" ? "ε" : "xᵀθ + α√…"}</span>;
}
