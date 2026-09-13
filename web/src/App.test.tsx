import { render, screen } from "@testing-library/react";
import { HashRouter } from "react-router-dom";

import { App } from "./App";

describe.each([
  ["#/", "See how bandit algorithms choose with incomplete information."],
  ["#/lesson/epsilon-greedy", "The Three Ancient Gates"],
  ["#/lesson/linucb", "The Labyrinth of Signals"],
  ["#/lab", "PyMAB Python Lab"],
])("route %s", (hash, heading) => {
  it(`renders ${heading}`, () => {
    window.location.hash = hash;
    render(
      <HashRouter>
        <App />
      </HashRouter>,
    );
    expect(screen.getByRole("heading", { level: 1, name: heading })).toBeInTheDocument();
  });
});

it("uses the Machine Face companion mark in the Arcade header", () => {
  const { container } = render(
    <HashRouter>
      <App />
    </HashRouter>,
  );

  expect(container.querySelector("img.brand-mark")).toHaveAttribute("src", "/pymab-mark.svg");
});
