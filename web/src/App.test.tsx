import { render, screen } from "@testing-library/react";
import { HashRouter } from "react-router-dom";

import { App } from "./App";

describe.each([
  ["#/", "Learn the art of choosing before you know."],
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
