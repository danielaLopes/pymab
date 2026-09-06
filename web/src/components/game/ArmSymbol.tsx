import type { LessonSnapshot } from "../../engine/protocol";

type Arm = LessonSnapshot["presentation"]["arms"][number];

const glyphs: Partial<Record<Arm["symbolKind"], string>> = {
  moon: "☾",
  sun: "☼",
  article: "▤",
  product: "▣",
  tutorial: "▶",
  allow: "✓",
  "light-check": "◒",
  "strong-verification": "⬡",
};

export function ArmSymbol({ arm }: { arm: Arm }) {
  return (
    <span
      className={`arm-symbol arm-symbol-${arm.symbolKind}`}
      data-symbol-kind={arm.symbolKind}
      aria-hidden="true"
    >
      {arm.symbolKind === "star" ? (
        <svg className="slender-star" viewBox="0 0 100 100">
          <polygon points="50,2 56,42 78,22 60,46 98,50 60,54 78,78 56,58 50,98 44,58 22,78 40,54 2,50 40,46 22,22 44,42" />
        </svg>
      ) : (
        (glyphs[arm.symbolKind] ?? "•")
      )}
    </span>
  );
}
