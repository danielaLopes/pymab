import { render } from "@testing-library/react";

import { ArmSymbol } from "./ArmSymbol";

describe("ArmSymbol", () => {
  it.each([
    ["article", "▤"],
    ["product", "▣"],
    ["tutorial", "▶"],
  ] as const)("uses the %s symbol selected by a recommendation candidate", (symbolKind, glyph) => {
    const { container } = render(
      <ArmSymbol arm={{ name: symbolKind, shortName: symbolKind, symbolKind }} />,
    );

    expect(container.querySelector(`[data-symbol-kind="${symbolKind}"]`)).toHaveTextContent(glyph);
  });

  it("uses the slender Christmas star instead of the relic glyph for Star Path", () => {
    const { container } = render(
      <ArmSymbol arm={{ name: "Star Path", shortName: "Star", symbolKind: "star" }} />,
    );

    const symbol = container.querySelector('[data-symbol-kind="star"]');
    expect(symbol?.querySelector(".slender-star polygon")).toBeInTheDocument();
    expect(symbol).not.toHaveTextContent("✦");
  });
});
