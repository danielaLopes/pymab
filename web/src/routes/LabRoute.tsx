import { basicSetup } from "codemirror";
import { python } from "@codemirror/lang-python";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { defaultKeymap, indentWithTab } from "@codemirror/commands";
import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";

import { useRuntime } from "../engine/RuntimeProvider";
import { LabClient } from "../lab/LabClient";
import type { LabResult } from "../lab/LabClient";

const epsilonExample = `import numpy as np
from pymab.policies import EpsilonGreedyPolicy

policy = EpsilonGreedyPolicy(n_arms=3, epsilon=0.2)
rng = np.random.default_rng(42)
for step in range(8):
    action = policy.select_action(rng=rng)
    reward = float(rng.random() < [0.25, 0.50, 0.75][action])
    policy.update(action=action, reward=reward)
    print(step + 1, "gate", action + 1, "reward", int(reward))
print("estimates:", policy.estimates)
`;

const linucbExample = `import numpy as np
from pymab.policies import LinUCBPolicy

policy = LinUCBPolicy(n_arms=3, n_features=4, alpha=1.0, l2=1.0)
rng = np.random.default_rng(31415)
for step in range(8):
    feature = np.r_[1.0, rng.choice([-1.0, 1.0], size=3)]
    context = np.repeat(feature[None, :], 3, axis=0)
    action = policy.select_action(context=context, rng=rng)
    reward = float(rng.random() < 0.5)
    policy.update(action=action, reward=reward, context=context)
    print(step + 1, "signals", feature[1:], "gate", action + 1)
`;

export function LabRoute() {
  const location = useLocation();
  const handedCode = (location.state as { code?: string } | null)?.code;
  const initialCode = handedCode ?? epsilonExample;
  const editorHost = useRef<HTMLDivElement>(null);
  const editor = useRef<EditorView | null>(null);
  const lab = useRef(new LabClient());
  const { client: lessonClient } = useRuntime();
  const [status, setStatus] = useState("Ready to run in a clean browser worker.");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<LabResult | null>(null);

  useEffect(() => {
    lessonClient.dispose();
    return () => lessonClient.restart();
  }, [lessonClient]);

  useEffect(() => {
    if (!editorHost.current) return;
    editor.current = new EditorView({
      state: EditorState.create({
        doc: initialCode,
        extensions: [
          basicSetup,
          keymap.of([...defaultKeymap, indentWithTab]),
          python(),
          EditorView.lineWrapping,
        ],
      }),
      parent: editorHost.current,
    });
    const labClient = lab.current;
    labClient.onProgress(setStatus);
    return () => {
      editor.current?.destroy();
      labClient.dispose();
    };
  }, [initialCode]);

  const setCode = (code: string) => {
    const view = editor.current;
    if (!view) return;
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: code } });
    setResult(null);
  };

  const run = async () => {
    const code = editor.current?.state.doc.toString() ?? "";
    setRunning(true);
    setResult(null);
    setStatus("Starting a clean Python process…");
    try {
      const next = await lab.current.run(code);
      setResult(next);
      setStatus(next.status === "success" ? "Run complete." : `Run ended: ${next.status}.`);
    } catch (error) {
      setResult({ status: "runtime", stdout: "", stderr: String(error), truncated: false });
      setStatus("The Lab could not start.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <main className="lab-page">
      <header className="mission-header">
        <Link to="/">← Mission map</Link>
        <p className="eyebrow">Developer workspace</p>
        <h1>PyMAB Python Lab</h1>
        <p>
          Edit and run real PyMAB code entirely inside your browser. Every stopped or timed-out run
          is discarded.
        </p>
      </header>
      <div className="lab-toolbar">
        <div>
          <button onClick={() => setCode(epsilonExample)}>ε-greedy example</button>
          <button onClick={() => setCode(linucbExample)}>LinUCB example</button>
        </div>
        <div>
          <button className="primary-button" disabled={running} onClick={() => void run()}>
            Run Python
          </button>
          <button
            disabled={!running}
            onClick={() => {
              lab.current.stop();
              setRunning(false);
              setStatus("Run stopped.");
            }}
          >
            Stop
          </button>
          <button disabled={running} onClick={() => setCode(initialCode)}>
            Reset
          </button>
        </div>
      </div>
      <section className="lab-workspace">
        <div>
          <h2>Code</h2>
          <div className="editor-shell" ref={editorHost} aria-label="Python code editor" />
        </div>
        <div className="console-panel">
          <h2>Output</h2>
          <p className="lab-status" role="status">
            {status}
          </p>
          {result?.stdout && (
            <>
              <h3>stdout</h3>
              <pre>{result.stdout}</pre>
            </>
          )}
          {(result?.stderr || result?.error) && (
            <>
              <h3>stderr</h3>
              <pre className="stderr">{result.stderr || result.error}</pre>
            </>
          )}
          {result?.truncated && <p>Output was truncated at 64 KiB.</p>}
        </div>
      </section>
    </main>
  );
}
