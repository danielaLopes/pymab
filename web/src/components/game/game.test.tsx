import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { CampaignMap, Chamber } from ".";

describe("Infinite Crossroads components", () => {
  it("gives every persistent gate a non-colour identity", () => {
    render(<Chamber snapshot={null} />);
    expect(screen.getByRole("button", { name: /Moon Gate, Memory/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Sun Gate, Promise/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Star Gate, Possibility/ })).toBeEnabled();
  });

  it("provides one route for every concrete policy", () => {
    render(<CampaignMap />, { wrapper: MemoryRouter });
    expect(screen.getAllByRole("link")).toHaveLength(27);
    expect(screen.getByRole("status")).toHaveTextContent("27 policies");
  });
});
