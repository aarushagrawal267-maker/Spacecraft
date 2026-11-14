import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type {
  Channel,
  DatasetStatus,
  DetectionDetail,
  DetectionRow,
  Experiment,
  MethodsResponse,
  Overview,
  RunDetail,
  Series,
  SystemInfo,
  Telemetry,
} from "./types";

export const useOverview = () =>
  useQuery({ queryKey: ["overview"], queryFn: () => api.get<Overview>("/overview"), refetchInterval: 10_000 });

export const useDatasetStatus = (poll = false) =>
  useQuery({
    queryKey: ["dataset-status"],
    queryFn: () => api.get<DatasetStatus>("/dataset/status"),
    refetchInterval: poll ? 1500 : false,
  });

export const useSystem = () => useQuery({ queryKey: ["system"], queryFn: () => api.get<SystemInfo>("/system"), staleTime: 60_000 });

export const useMethods = () =>
  useQuery({ queryKey: ["methods"], queryFn: () => api.get<MethodsResponse>("/methods"), staleTime: Infinity });

export const useChannels = () =>
  useQuery({ queryKey: ["channels"], queryFn: () => api.get<Channel[]>("/channels"), staleTime: 15_000 });

export const useTelemetry = (chan: string | null, split: "train" | "test", maxPoints?: number) =>
  useQuery({
    queryKey: ["telemetry", chan, split, maxPoints],
    queryFn: () => api.get<Telemetry>(`/telemetry/${chan}?split=${split}${maxPoints ? `&max_points=${maxPoints}` : ""}`),
    enabled: !!chan,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  });

export const useExperiments = (poll = true) =>
  useQuery({
    queryKey: ["experiments"],
    queryFn: () => api.get<Experiment[]>("/experiments"),
    refetchInterval: poll ? 4000 : false,
  });

export const useExperiment = (id: number | null, poll = false) =>
  useQuery({
    queryKey: ["experiment", id],
    queryFn: () => api.get<Experiment>(`/experiments/${id}`),
    enabled: id != null,
    refetchInterval: (q) => (poll && q.state.data && ["queued", "running"].includes(q.state.data.status) ? 2000 : false),
    placeholderData: keepPreviousData,
  });

export const useRun = (id: number | null) =>
  useQuery({ queryKey: ["run", id], queryFn: () => api.get<RunDetail>(`/runs/${id}`), enabled: id != null });

export const useSeries = (id: number | null) =>
  useQuery({
    queryKey: ["series", id],
    queryFn: () => api.get<Series>(`/runs/${id}/series`),
    enabled: id != null,
    staleTime: Infinity,
    retry: false,
  });

export const useDetections = (params: { experiment_id?: number | null; chan_id?: string | null; outcome?: string | null }) => {
  const q = new URLSearchParams();
  if (params.experiment_id) q.set("experiment_id", String(params.experiment_id));
  if (params.chan_id) q.set("chan_id", params.chan_id);
  if (params.outcome) q.set("outcome", params.outcome);
  q.set("limit", "2000");
  return useQuery({
    queryKey: ["detections", params],
    queryFn: () => api.get<DetectionRow[]>(`/detections?${q}`),
    placeholderData: keepPreviousData,
  });
};

export const useDetection = (id: number | null) =>
  useQuery({
    queryKey: ["detection", id],
    queryFn: () => api.get<DetectionDetail>(`/detections/${id}?context=400`),
    enabled: id != null,
  });

export const useCompare = (ids: number[]) =>
  useQuery({
    queryKey: ["compare", ids],
    queryFn: () => api.get<{ experiments: Experiment[] }>(`/evaluation/compare?ids=${ids.join(",")}`),
    enabled: ids.length > 0,
    placeholderData: keepPreviousData,
  });

export function useCreateExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Record<string, unknown>) => api.post<Experiment>("/experiments", body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
      qc.invalidateQueries({ queryKey: ["overview"] });
    },
  });
}

export function useCancelExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.post<{ result: string }>(`/experiments/${id}/cancel`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["experiments"] }),
  });
}

export function useDeleteExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.del<{ deleted: number }>(`/experiments/${id}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
      qc.invalidateQueries({ queryKey: ["overview"] });
      qc.invalidateQueries({ queryKey: ["channels"] });
    },
  });
}
