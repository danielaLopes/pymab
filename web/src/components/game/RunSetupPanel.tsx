import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { policiesByFamily, policyCatalog, type PolicyId } from "@/catalog/policies";
import type { LessonMode } from "@/engine/protocol";
import type { EnvironmentDraft } from "@/state/environments";
import {
  configurationsMatch,
  formatParameter,
  modeLabels,
  validateRunDraft,
  type DraftParameterValue,
  type RunConfiguration,
  type RunDraftConfiguration,
} from "@/state/runConfiguration";
import { EnvironmentEditor } from "./EnvironmentEditor";

const lessonModes: LessonMode[] = ["guided", "challenge", "freePlay"];

export interface RunSetupPanelProps {
  activeConfiguration: RunConfiguration | null;
  draftConfiguration: RunDraftConfiguration;
  challengeTarget: string;
  pending: boolean;
  onPolicyChange: (policyId: PolicyId) => void;
  onModeChange: (mode: LessonMode) => void;
  onParameterChange: (key: string, value: DraftParameterValue) => void;
  onSeedChange: (value: string) => void;
  onEnvironmentChange: (environment: EnvironmentDraft) => void;
  onRegenerateEnvironment: () => void;
  onRestorePolicyDefaults: () => void;
  onApply: () => void;
}

function activeRunSummary(configuration: RunConfiguration | null): string {
  if (!configuration) return "No run has started.";
  const policy = policyCatalog[configuration.policyId];
  const parameters = Object.entries(configuration.parameters)
    .map(([key, value]) => `${key} ${value === null ? "default" : String(value)}`)
    .join(" · ");
  return `${policy.label} · ${modeLabels[configuration.mode]}${parameters ? ` · ${parameters}` : ""} · seed ${configuration.seed}`;
}

function ParameterField({
  definition,
  value,
  error,
  pending,
  onChange,
}: {
  definition: (typeof policyCatalog)[PolicyId]["parameters"][number];
  value: DraftParameterValue | undefined;
  error: string | undefined;
  pending: boolean;
  onChange: (value: DraftParameterValue) => void;
}) {
  const inputId = `run-parameter-${definition.key}`;
  const errorId = `${inputId}-error`;
  const helpId = `${inputId}-help`;

  if (definition.kind === "boolean") {
    return (
      <div className="run-setup-field parameter-card">
        <label className="boolean-parameter" htmlFor={inputId}>
          <span>
            <strong>{definition.label}</strong>
            <small>{definition.help}</small>
          </span>
          <input
            id={inputId}
            type="checkbox"
            checked={value === true}
            disabled={pending}
            onChange={(event) => onChange(event.target.checked)}
          />
        </label>
      </div>
    );
  }

  if (definition.kind === "select") {
    return (
      <div className="run-setup-field parameter-card">
        <Label htmlFor={inputId}>{definition.label}</Label>
        <Select
          value={typeof value === "string" ? value : ""}
          disabled={pending}
          onValueChange={onChange}
        >
          <SelectTrigger id={inputId} aria-describedby={error ? errorId : helpId}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {definition.options?.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p id={helpId} className="field-help">
          {definition.help}
        </p>
        {error && (
          <p id={errorId} className="field-error" role="alert">
            {error}
          </p>
        )}
      </div>
    );
  }

  const numeric = Number(value);
  const hasSlider =
    definition.kind !== "optional-number" &&
    definition.minimum !== undefined &&
    definition.maximum !== undefined &&
    definition.step !== undefined;
  const sliderValue = Number.isFinite(numeric)
    ? Math.min(definition.maximum ?? numeric, Math.max(definition.minimum ?? numeric, numeric))
    : (definition.minimum ?? 0);
  return (
    <div className="run-setup-field parameter-card">
      <div className="run-setup-label-row">
        <Label htmlFor={inputId}>{definition.label}</Label>
        {definition.shortLabel && <span>{definition.shortLabel}</span>}
      </div>
      <div className={hasSlider ? "parameter-inputs" : "parameter-inputs number-only"}>
        {hasSlider && (
          <Slider
            aria-label={`${definition.label} slider`}
            min={definition.minimum!}
            max={definition.maximum!}
            step={definition.step!}
            value={[sliderValue]}
            disabled={pending}
            onValueChange={(values) => {
              const next = values[0];
              if (next !== undefined) onChange(formatParameter(next));
            }}
          />
        )}
        <Input
          id={inputId}
          className="parameter-number"
          type="number"
          inputMode="decimal"
          min={definition.minimum}
          max={definition.maximum}
          step={definition.step}
          value={typeof value === "string" ? value : ""}
          placeholder={definition.kind === "optional-number" ? "Use default" : undefined}
          disabled={pending}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? errorId : helpId}
          onChange={(event) => onChange(event.target.value)}
        />
      </div>
      <p id={helpId} className="field-help">
        {definition.help}
      </p>
      {error && (
        <p id={errorId} className="field-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

export function RunSetupPanel({
  activeConfiguration,
  draftConfiguration,
  challengeTarget,
  pending,
  onPolicyChange,
  onModeChange,
  onParameterChange,
  onSeedChange,
  onEnvironmentChange,
  onRegenerateEnvironment,
  onRestorePolicyDefaults,
  onApply,
}: RunSetupPanelProps) {
  const policy = policyCatalog[draftConfiguration.policyId];
  const { configuration, errors } = validateRunDraft(draftConfiguration);
  const dirty = !configurationsMatch(draftConfiguration, activeConfiguration);
  const fixedSeed = draftConfiguration.mode !== "freePlay";
  const primaryParameters = policy.parameters.filter((item) => !item.advanced);
  const advancedParameters = policy.parameters.filter((item) => item.advanced);

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
        <div className="run-setup-field run-setup-policy">
          <Label htmlFor="policy-select">Policy</Label>
          <Select
            value={draftConfiguration.policyId}
            disabled={pending}
            onValueChange={(value) => onPolicyChange(value as PolicyId)}
          >
            <SelectTrigger id="policy-select" aria-label="Policy">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(policiesByFamily).map(([family, policies]) => (
                <SelectGroup key={family}>
                  <SelectLabel>{family.replace("-", " ")}</SelectLabel>
                  {policies.map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
          <p className="field-help">{policy.className}</p>
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

        <div className="parameter-section">
          <div className="parameter-section-heading">
            <div>
              <h3>Policy parameters</h3>
              <p>These values are passed to the public PyMAB constructor.</p>
            </div>
            <Button type="button" variant="outline" size="sm" onClick={onRestorePolicyDefaults}>
              Restore policy defaults
            </Button>
          </div>
          {primaryParameters.length ? (
            <div className="dynamic-parameter-grid">
              {primaryParameters.map((definition) => (
                <ParameterField
                  key={definition.key}
                  definition={definition}
                  value={draftConfiguration.parameters[definition.key]}
                  error={errors.parameters[definition.key]}
                  pending={pending}
                  onChange={(value) => onParameterChange(definition.key, value)}
                />
              ))}
            </div>
          ) : (
            <p className="no-parameters">This policy has no configurable constructor values.</p>
          )}
          {policy.id === "moss" && (
            <p className="constructor-note">
              <code>
                horizon=
                {draftConfiguration.mode === "guided" ? policy.horizon : policy.challengeHorizon}
              </code>{" "}
              is fixed to the run length.
            </p>
          )}
          {advancedParameters.length > 0 && (
            <details className="advanced-parameters">
              <summary>Advanced numerical settings</summary>
              <div className="dynamic-parameter-grid">
                {advancedParameters.map((definition) => (
                  <ParameterField
                    key={definition.key}
                    definition={definition}
                    value={draftConfiguration.parameters[definition.key]}
                    error={errors.parameters[definition.key]}
                    pending={pending}
                    onChange={(value) => onParameterChange(definition.key, value)}
                  />
                ))}
              </div>
            </details>
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

        {draftConfiguration.mode === "freePlay" && (
          <EnvironmentEditor
            draft={draftConfiguration.environment}
            errors={errors.environment}
            source={draftConfiguration.probabilitySource}
            pending={pending}
            onChange={onEnvironmentChange}
            onRegenerate={onRegenerateEnvironment}
          />
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
