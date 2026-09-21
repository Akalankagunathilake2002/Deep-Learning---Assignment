// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import App from "./App.jsx";
import examples from "./examples.json";
import { GruModel } from "./lib/gru.js";
import fixtures from "./lib/__fixtures__/fixtures.json";

// The real exported models and saved results, loaded from disk instead of over HTTP.
const publicDir = join(process.cwd(), "public"); // tests run from the frontend folder
const json = (path) => JSON.parse(readFileSync(join(publicDir, path), "utf8"));
const arrayBuffer = (path) => {
  const b = readFileSync(join(publicDir, path));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const config = json("model/config.json");
const engine = {
  config,
  vocab: new Map(Object.entries(json("model/vocab.json"))),
  models: config.models.map((m) => ({
    id: m.id,
    name: m.name,
    model: new GruModel(json(`model/${m.dir}/manifest.json`), arrayBuffer(`model/${m.dir}/weights.bin`)),
  })),
};
const results = json("results.json");
const load = () => Promise.resolve([engine, results]);

const expectedLabel = (text, id) => (fixtures.find((f) => f.text === text)[id] >= 0.5 ? "Positive" : "Negative");
const example = (id) => examples.find((e) => e.id === id);

afterEach(cleanup);

async function openApp() {
  render(<App load={load} />);
  await screen.findByRole("button", { name: "Positive" }); // examples are enabled once the models have loaded
}

describe("Try it", () => {
  it("disables Analyze until there is text", async () => {
    await openApp();
    const analyze = screen.getByRole("button", { name: "Analyze review" });
    expect(analyze).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Hotel review"), { target: { value: "Nice room and friendly staff." } });
    expect(analyze).toBeEnabled();
  });

  it("shows an answer from both models when an example is picked", async () => {
    await openApp();
    fireEvent.click(screen.getByRole("button", { name: "Positive" }));
    const text = example("positive").text;
    expect(screen.getByLabelText("Hotel review")).toHaveValue(text);
    for (const [id, name] of [["main", "Main GRU"], ["short", "GRU + short examples"]]) {
      expect(await screen.findByRole("article", { name: `${name}: ${expectedLabel(text, id)}` })).toBeInTheDocument();
    }
    expect(screen.getAllByRole("meter")).toHaveLength(2);
  });

  it("agrees with Keras for a negative review too", async () => {
    await openApp();
    fireEvent.click(screen.getByRole("button", { name: "Negative" }));
    const text = example("negative").text;
    expect(await screen.findByRole("article", { name: `Main GRU: ${expectedLabel(text, "main")}` })).toBeInTheDocument();
    expect(expectedLabel(text, "main")).toBe("Negative");
  });

  it("warns about very short reviews", async () => {
    await openApp();
    fireEvent.click(screen.getByRole("button", { name: "Short negative" }));
    expect(await screen.findByText(/Short review\./)).toBeInTheDocument();
  });

  it("says when nothing usable is left after cleaning", async () => {
    await openApp();
    fireEvent.change(screen.getByLabelText("Hotel review"), { target: { value: "??? 123" } });
    fireEvent.click(screen.getByRole("button", { name: "Analyze review" }));
    expect(await screen.findByText(/No usable words were found/)).toBeInTheDocument();
    expect(screen.queryAllByRole("article")).toHaveLength(0);
  });

  it("tells you when the text was edited after the last analysis", async () => {
    await openApp();
    fireEvent.click(screen.getByRole("button", { name: "Positive" }));
    await screen.findByRole("article", { name: /Main GRU/ });
    fireEvent.change(screen.getByLabelText("Hotel review"), { target: { value: "Something else entirely." } });
    expect(screen.getByText(/You have edited the text since this result/)).toBeInTheDocument();
  });

  it("marks words that are not in the vocabulary", async () => {
    await openApp();
    fireEvent.change(screen.getByLabelText("Hotel review"), { target: { value: "The room was wonderful but zxqvbn was odd." } });
    fireEvent.click(screen.getByRole("button", { name: "Analyze review" }));
    fireEvent.click(await screen.findByText("What the models saw"));
    expect(screen.getByText(/1 not in the vocabulary/)).toBeInTheDocument();
  });

  it("Clear empties the text and the result", async () => {
    await openApp();
    fireEvent.click(screen.getByRole("button", { name: "Positive" }));
    await screen.findByRole("article", { name: /Main GRU/ });
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByLabelText("Hotel review")).toHaveValue("");
    expect(screen.queryAllByRole("article")).toHaveLength(0);
  });
});

describe("Results & about", () => {
  it("shows the saved test-set results", async () => {
    await openApp();
    fireEvent.click(screen.getByRole("tab", { name: "Results & about" }));
    expect(await screen.findByRole("heading", { name: "How well it works" })).toBeInTheDocument();
    expect(screen.getAllByText(`${(results.main.accuracy * 100).toFixed(1)}%`).length).toBeGreaterThan(0);
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getAllByRole("img").length).toBe(4);
  });

  it("switches tabs with the arrow keys", async () => {
    await openApp();
    fireEvent.keyDown(screen.getByRole("tab", { name: "Try it" }), { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Results & about" })).toHaveAttribute("aria-selected", "true");
  });
});

describe("when the models cannot be loaded", () => {
  it("shows a clear message instead of a broken page", async () => {
    render(<App load={() => Promise.reject(new Error("Could not load model/config.json (HTTP 404)."))} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load model/config.json (HTTP 404).");
    expect(screen.getByRole("button", { name: "Analyze review" })).toBeDisabled();
  });
});

describe("Sidebar and Model Selection", () => {
  it("renders all four model buttons with GRU initially selected", async () => {
    await openApp();
    expect(screen.getByRole("button", { name: /RNN/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /LSTM/ })).toBeInTheDocument();
    const gruBtn = screen.getByRole("button", { name: /GRU/ });
    expect(gruBtn).toBeInTheDocument();
    expect(gruBtn).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: /1D CNN/ })).toBeInTheDocument();
  });

  it("switches to RNN profile and shows Under Development status", async () => {
    await openApp();
    fireEvent.click(screen.getByRole("button", { name: /RNN/ }));
    expect(await screen.findByRole("heading", { name: "Simple RNN" })).toBeInTheDocument();
    expect(screen.getByText("Under Development")).toBeInTheDocument();
    expect(screen.getByText("SimpleRNN(64)")).toBeInTheDocument();
    expect(screen.queryByLabelText("Hotel review")).not.toBeInTheDocument();
  });

  it("switches to LSTM profile and 1D CNN profile", async () => {
    await openApp();
    fireEvent.click(screen.getByRole("button", { name: /LSTM/ }));
    expect(await screen.findByRole("heading", { name: "LSTM" })).toBeInTheDocument();
    expect(screen.getByText("LSTM(64)")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /1D CNN/ }));
    expect(await screen.findByRole("heading", { name: "1D CNN" })).toBeInTheDocument();
    expect(screen.getByText("Conv1D(64, kernel_size=3)")).toBeInTheDocument();
  });

  it("returns to GRU and restores the live sentiment analyzer", async () => {
    await openApp();
    fireEvent.click(screen.getByRole("button", { name: /LSTM/ }));
    expect(screen.queryByLabelText("Hotel review")).not.toBeInTheDocument();

    const switchBtn = screen.getByRole("button", { name: /Try Live GRU Model/ });
    fireEvent.click(switchBtn);

    expect(await screen.findByLabelText("Hotel review")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Try it" })).toBeInTheDocument();
  });
});

