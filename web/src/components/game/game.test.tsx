import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { CampaignMap, Chamber, ModeTabs } from ".";

describe("Infinite Crossroads components", () => {
  it("gives every persistent gate a non-colour identity", () => {
    render(<Chamber snapshot={null} />);
    expect(screen.getByRole("button", { name: /Moon Gate, Memory/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Sun Gate, Promise/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Star Gate, Possibility/ })).toBeEnabled();
  });

  it("exposes lesson modes as pressed buttons", () => {
    render(<ModeTabs mode="guided" onChange={() => undefined} />);
    expect(screen.getByRole("button", { name: "Guided" })).toHaveAttribute("aria-pressed", "true");
  });

  it("provides two navigable campaign missions", () => {
    render(<CampaignMap />, { wrapper: MemoryRouter });
    expect(screen.getAllByRole("link")).toHaveLength(2);
  });
});
