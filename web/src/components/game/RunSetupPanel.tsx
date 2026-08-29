import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { LessonId, LessonMode } from "@/engine/protocol";
import {
  algorithmLabels,
  configurationsMatch,
  formatParameter,
  modeLabels,
  parameterDefinitions,
  validateRunDraft,
  type RunConfiguration,
  type RunDraftConfiguration,
} from "@/state/runConfiguration";
import { formatProbabilityPercent } from "@/state/portalProbabilities";

const lessonIds: LessonId[] = ["epsilon-greedy", "linucb"];
const lessonModes: LessonMode[] = ["guided", "challenge", "freePlay"];
const portalNames = ["Moon", "Sun", "Star"] as const;

export interface RunSetupPanelProps {
  activeConfiguration: RunConfiguration | null;
  draftConfiguration: RunDraftConfiguration;
  challengeTarget: string;
  pending: boolean;
  onAlgorithmChange: (lessonId: LessonId) => void;
  onModeChange: (mode: LessonMode) => void;
  onParameterChange: (value: string) => void;
  onSeedChange: (value: string) => void;
  onPortalProbabilityChange: (index: number, value: string) => void;
  onUseSeedGeneratedValues: () => void;
  onApply: () => void;
}

function activeRunSummary(configuration: RunConfiguration | null): string {
  if (!configuration) return "No run has started.";
  const definition = parameterDefinitions[configuration.lessonId];
  return `${algorithmLabels[configuration.lessonId]} · ${modeLabels[configuration.mode]} · ${definition.shortLabel} ${configuration.parameter} · seed ${configuration.seed}`;
}

export function RunSetupPanel({
  activeConfiguration,
  draftConfiguration,
  challengeTarget,
  pending,
  onAlgorithmChange,
  onModeChange,
  onParameterChange,
  onSeedChange,
  onPortalProbabilityChange,
  onUseSeedGeneratedValues,
  onApply,
}: RunSetupPanelProps) {
  const definition = parameterDefinitions[draftConfiguration.lessonId];
  const { configuration, errors } = validateRunDraft(draftConfiguration);
  const dirty = !configurationsMatch(draftConfiguration, activeConfiguration);
  const numericParameter = Number(draftConfiguration.parameter);
  const sliderValue = Number.isFinite(numericParameter)
    ? Math.min(definition.maximum, Math.max(definition.minimum, numericParameter))
    : definition.defaultValue;
  const fixedSeed = draftConfiguration.mode !== "freePlay";
  const showPortalProbabilities =
    draftConfiguration.lessonId === "epsilon-greedy" && draftConfiguration.mode === "freePlay";

  return (
    <section className="run-setup" aria-labelledby="run-setup-title">
      <div className="run-setup-heading">
        <div>
          <p className="eyebrow">Run setup</p>
          <h2 id="run-setup-title">Configure this run</h2>
        </div>
        <p>Changes take effect when you start the run.</p>
      </div>

      <div className="run-setup-grid">
        <div className="run-setup-field run-setup-selector">
          <span className="run-setup-label" id="algorithm-label">
            Algorithm
          </span>
          <ToggleGroup
            type="single"
            value={draftConfiguration.lessonId}
            disabled={pending}
            aria-labelledby="algorithm-label"
            onValueChange={(value) => {
              if (value) onAlgorithmChange(value as LessonId);
            }}
          >
            {lessonIds.map((lessonId) => (
              <ToggleGroupItem key={lessonId} value={lessonId}>
                {algorithmLabels[lessonId]}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>

        <div className="run-setup-field run-setup-selector">
          <span className="run-setup-label" id="mode-label">
            Run mode
          </span>
          <ToggleGroup
            type="single"
            value={draftConfiguration.mode}
            disabled={pending}
            aria-labelledby="mode-label"
            onValueChange={(value) => {
              if (value) onModeChange(value as LessonMode);
            }}
          >
            {lessonModes.map((mode) => (
              <ToggleGroupItem key={mode} value={mode}>
                {modeLabels[mode]}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          {draftConfiguration.mode === "challenge" && (
            <p className="mode-target">{challengeTarget}</p>
          )}
        </div>

        <div className="run-setup-field run-setup-parameter">
          <div className="run-setup-label-row">
            <Label htmlFor="run-parameter">{definition.label}</Label>
            <span>{definition.shortLabel}</span>
          </div>
          <div className="parameter-inputs">
            <Slider
              aria-label={`${definition.label} slider`}
              min={definition.minimum}
              max={definition.maximum}
              step={definition.step}
              value={[sliderValue]}
              disabled={pending}
              onValueChange={(values) => {
                const value = values[0];
                if (value !== undefined) onParameterChange(formatParameter(value));
              }}
            />
            <Input
              id="run-parameter"
              className="parameter-number"
              type="number"
              inputMode="decimal"
              min={definition.minimum}
              max={definition.maximum}
              step={definition.step}
              value={draftConfiguration.parameter}
              disabled={pending}
              aria-invalid={Boolean(errors.parameter)}
              aria-describedby={errors.parameter ? "parameter-error" : "parameter-range"}
              onChange={(event) => onParameterChange(event.target.value)}
            />
          </div>
          <p id="parameter-range" className="field-help">
            {definition.minimum} to {definition.maximum}, step {definition.step}
          </p>
          {errors.parameter && (
            <p id="parameter-error" className="field-error" role="alert">
              {errors.parameter}
            </p>
          )}
        </div>

        <div className="run-setup-field run-setup-seed">
          <div className="run-setup-label-row">
            <Label htmlFor="run-seed">Random seed</Label>
            {fixedSeed && <span>Fixed for {modeLabels[draftConfiguration.mode]}</span>}
          </div>
          <Input
            id="run-seed"
            type="number"
            inputMode="numeric"
            step={1}
            value={draftConfiguration.seed}
            readOnly={fixedSeed}
            disabled={pending}
            aria-invalid={Boolean(errors.seed)}
            aria-describedby={errors.seed ? "seed-error" : "seed-help"}
            onChange={(event) => onSeedChange(event.target.value)}
          />
          <p id="seed-help" className="field-help">
            {fixedSeed ? "This mode uses a repeatable seed." : "Use any safe whole number."}
          </p>
          {errors.seed && (
            <p id="seed-error" className="field-error" role="alert">
              {errors.seed}
            </p>
          )}
        </div>

        {showPortalProbabilities && (
          <fieldset className="portal-probabilities">
            <legend>Portal relic chances</legend>
            <div className="portal-probabilities-heading">
              <p>Set the chance that each portal contains a relic.</p>
              <div className="probability-source">
                <span>
                  {draftConfiguration.probabilitySource === "generated"
                    ? "Generated from seed"
                    : "Custom"}
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={pending || Boolean(errors.seed)}
                  onClick={onUseSeedGeneratedValues}
                >
                  Use seed-generated values
                </Button>
              </div>
            </div>
            <div className="portal-probability-grid">
              {portalNames.map((name, index) => {
                const value = draftConfiguration.portalProbabilities[index];
                const numericValue = Number(value);
                const sliderProbability = Number.isFinite(numericValue)
                  ? Math.min(100, Math.max(0, numericValue))
                  : 50;
                const error = errors.portalProbabilities?.[index];
                const inputId = `portal-probability-${index}`;
                const errorId = `${inputId}-error`;
                return (
                  <div className="portal-probability-card" key={name}>
                    <div className="run-setup-label-row">
                      <Label htmlFor={inputId}>{name} Gate</Label>
                      <span>{formatProbabilityPercent(sliderProbability / 100)}%</span>
                    </div>
                    <div className="parameter-inputs">
                      <Slider
                        aria-label={`${name} Gate relic chance slider`}
                        min={0}
                        max={100}
                        step={0.1}
                        value={[sliderProbability]}
                        disabled={pending}
                        onValueChange={(values) => {
                          const next = values[0];
                          if (next !== undefined) {
                            onPortalProbabilityChange(index, formatProbabilityPercent(next / 100));
                          }
                        }}
                      />
                      <div className="probability-number-wrap">
                        <Input
                          id={inputId}
                          className="parameter-number"
                          type="number"
                          inputMode="decimal"
                          min={0}
                          max={100}
                          step={0.1}
                          value={value}
                          disabled={pending}
                          aria-invalid={Boolean(error)}
                          aria-describedby={error ? errorId : undefined}
                          onChange={(event) => onPortalProbabilityChange(index, event.target.value)}
                        />
                        <span aria-hidden="true">%</span>
                      </div>
                    </div>
                    {error && (
                      <p id={errorId} className="field-error" role="alert">
                        {error}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </fieldset>
        )}
      </div>

      <div className="run-setup-footer">
        <div className="current-run">
          <span>Current run</span>
          <strong>{activeRunSummary(activeConfiguration)}</strong>
          <p className={dirty ? "draft-status dirty" : "draft-status"} aria-live="polite">
            {dirty ? "Changes have not been applied." : "These settings match the current run."}
          </p>
        </div>
        <Button type="button" disabled={pending || configuration === null} onClick={onApply}>
          {pending
            ? "Starting run..."
            : activeConfiguration
              ? "Restart with these settings"
              : "Start new run"}
        </Button>
      </div>
    </section>
  );
}
