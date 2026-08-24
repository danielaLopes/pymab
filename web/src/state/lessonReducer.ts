import type { LessonMode, LessonSnapshot } from "../engine/protocol";

export type LessonPhase =
  | "loading"
  | "guided"
  | "challengeSetup"
  | "challengeRunning"
  | "debrief"
  | "freePlay"
  | "error"
  | "unsupported";

export interface LessonState {
  phase: LessonPhase;
  mode: LessonMode;
  snapshot: LessonSnapshot | null;
  pending: boolean;
  error: string | null;
}

export type LessonAction =
  | { type: "loading" }
  | { type: "started"; mode: LessonMode; snapshot: LessonSnapshot }
  | { type: "pending"; value: boolean }
  | { type: "snapshot"; snapshot: LessonSnapshot }
  | { type: "error"; message: string }
  | { type: "unsupported"; message: string }
  | { type: "challengeSetup" };

export const initialLessonState: LessonState = {
  phase: "loading",
  mode: "guided",
  snapshot: null,
  pending: false,
  error: null,
};

export function lessonReducer(state: LessonState, action: LessonAction): LessonState {
  switch (action.type) {
    case "loading":
      return { ...initialLessonState };
    case "started":
      return {
        phase:
          action.mode === "guided"
            ? "guided"
            : action.mode === "challenge"
              ? "challengeRunning"
              : "freePlay",
        mode: action.mode,
        snapshot: action.snapshot,
        pending: false,
        error: null,
      };
    case "pending":
      return { ...state, pending: action.value };
    case "snapshot":
      return {
        ...state,
        phase: action.snapshot.completed ? "debrief" : state.phase,
        snapshot: action.snapshot,
        pending: false,
      };
    case "challengeSetup":
      return { ...state, phase: "challengeSetup", pending: false };
    case "error":
      return { ...state, phase: "error", pending: false, error: action.message };
    case "unsupported":
      return { ...state, phase: "unsupported", pending: false, error: action.message };
  }
}
