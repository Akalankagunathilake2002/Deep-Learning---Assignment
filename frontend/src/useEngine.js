import { useEffect, useState } from "react";
import { loadEngine, loadResults } from "./lib/engine.js";

/** Loads the models and the saved results once. Returns { status: "loading" | "ready" | "error", ... }. */
export function useEngine(load = () => Promise.all([loadEngine(), loadResults()])) {
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    load()
      .then(([engine, results]) => !cancelled && setState({ status: "ready", engine, results }))
      .catch((error) => !cancelled && setState({ status: "error", message: error.message }));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return state;
}
