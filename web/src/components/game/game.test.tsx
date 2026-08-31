import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { DecisionHistoryBoard } from "./DecisionHistoryBoard";
import { CampaignMap } from ".";

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
});
