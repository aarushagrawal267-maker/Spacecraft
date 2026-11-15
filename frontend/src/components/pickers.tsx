import { useMemo } from "react";
import { useChannels } from "../api/hooks";
import type { Channel } from "../api/types";
import { METHOD_ORDER } from "../lib/palette";
import { Segmented, Select } from "./ui";

export function ChannelPicker({
  chan,
  onChange,
  spacecraft,
  onSpacecraft,
}: {
  chan: string;
  onChange: (c: string) => void;
  spacecraft?: "SMAP" | "MSL";
  onSpacecraft?: (s: "SMAP" | "MSL") => void;
}) {
  const { data } = useChannels();
  const current = data?.find((c) => c.chan_id === chan);
  const sc = spacecraft ?? current?.spacecraft ?? "SMAP";
  const list = (data ?? []).filter((c) => c.spacecraft === sc);
  return (
    <div className="flex items-center gap-2">
      <Segmented
        ariaLabel="Spacecraft"
        value={sc}
        options={[
          { value: "SMAP", label: "SMAP" },
          { value: "MSL", label: "MSL · Curiosity" },
        ]}
        onChange={(v) => {
          onSpacecraft?.(v);
          const first = (data ?? []).find((c) => c.spacecraft === v);
          if (first && current?.spacecraft !== v) onChange(first.chan_id);
        }}
      />
      <Select ariaLabel="Telemetry channel" value={chan} onChange={onChange} className="num w-[150px]">
        {list.map((c) => (
          <option key={c.chan_id} value={c.chan_id}>
            {c.chan_id} · {c.n_intervals} label{c.n_intervals === 1 ? "" : "s"}
          </option>
        ))}
      </Select>
    </div>
  );
}

export function useChannel(chan: string): Channel | undefined {
  const { data } = useChannels();
  return data?.find((c) => c.chan_id === chan);
}

/** Completed runs for a channel, newest first per method (reference and local). */
export function useRunsForChannel(chan: string) {
  const ch = useChannel(chan);
  return useMemo(() => {
    const runs = ch?.runs ?? [];
    return [...runs].sort(
      (a, b) => METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method) || b.run_id - a.run_id,
    );
  }, [ch]);
}

export function RunPicker({
  chan,
  runId,
  onChange,
  requireModel = false,
}: {
  chan: string;
  runId: number | null;
  onChange: (id: number) => void;
  requireModel?: boolean;
}) {
  const runs = useRunsForChannel(chan).filter((r) => !requireModel || r.has_model || isStateless(r.method));
  if (!runs.length) {
    return <span className="text-[12.5px] text-muted">No trained runs for {chan}</span>;
  }
  return (
    <Select ariaLabel="Trained run" value={String(runId ?? "")} onChange={(v) => onChange(Number(v))} className="max-w-[380px]">
      {runs.map((r) => (
        <option key={r.run_id} value={r.run_id}>
          {r.experiment_name} (exp {r.experiment_id})
        </option>
      ))}
    </Select>
  );
}

export const isStateless = (m: string) => ["persistence", "rolling_zscore", "random"].includes(m);
