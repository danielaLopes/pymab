import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import type { LessonSnapshot } from "../../engine/protocol";
import { DecisionHistoryBoard } from "./DecisionHistoryBoard";
import { CampaignMap, PolicyBars } from ".";

describe("Infinite Crossroads components", () => {
  it("gives every path a persistent non-colour identity", () => {
    render(<DecisionHistoryBoard snapshot={null} />);
    expect(screen.getByRole("columnheader", { name: /Moon Path/ })).toBeVisible();
    expect(screen.getByRole("columnheader", { name: /Sun Path/ })).toBeVisible();
    expect(screen.getByRole("columnheader", { name: /Star Path/ })).toBeVisible();
    expect(screen.getByRole("row", { name: /awaiting the first decision/i })).toBeVisible();
  });

  it("provides one route for every concrete policy", () => {
    render(<CampaignMap />, { wrapper: MemoryRouter });
    expect(screen.getAllByRole("link")).toHaveLength(27);
    expect(screen.getByRole("status")).toHaveTextContent("27 policies");
  });

  it("uses recommendation candidate symbols in the predicted probability state", () => {
    const snapshot = {
      policyId: "logistic-contextual-bandit",
      presentation: {
        arms: [
          { name: "Article", shortName: "Article", symbolKind: "article" },
          { name: "Product", shortName: "Product", symbolKind: "product" },
          { name: "Tutorial", shortName: "Tutorial", symbolKind: "tutorial" },
        ],
      },
      diagnostic: {
        decision: {
          label: "Predicted click probability",
          values: [0.2, 0.5, 0.8],
        },
      },
    } as unknown as LessonSnapshot;

    const { container } = render(<PolicyBars snapshot={snapshot} />);

    expect(container.querySelector('[data-symbol-kind="article"]')).toHaveTextContent("▤");
    expect(container.querySelector('[data-symbol-kind="product"]')).toHaveTextContent("▣");
    expect(container.querySelector('[data-symbol-kind="tutorial"]')).toHaveTextContent("▶");
    expect(container.querySelector('[data-symbol-kind="moon"]')).not.toBeInTheDocument();
  });
});
