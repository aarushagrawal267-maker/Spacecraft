import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

interface Selection {
  chan: string;
  setChan: (c: string) => void;
  method: string;
  setMethod: (m: string) => void;
}

const Ctx = createContext<Selection | null>(null);

function read(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}
function write(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable — selection simply isn't remembered */
  }
}

/** Remembers the last inspected channel and method across pages (per browser only). */
export function SelectionProvider({ children }: { children: ReactNode }) {
  const [chan, setChan] = useState(() => read("hg.chan", "P-1"));
  const [method, setMethod] = useState(() => read("hg.method", "lstm_forecaster"));
  useEffect(() => write("hg.chan", chan), [chan]);
  useEffect(() => write("hg.method", method), [method]);
  return <Ctx.Provider value={{ chan, setChan, method, setMethod }}>{children}</Ctx.Provider>;
}

export function useSelection() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useSelection outside provider");
  return v;
}
