import { Component, lazy, Suspense, type ReactNode } from "react";
import { Route, Routes } from "react-router-dom";
import { Shell } from "./components/Shell";
import { ErrorState, Loading } from "./components/ui";

const Overview = lazy(() => import("./pages/Overview"));
const TelemetryExplorer = lazy(() => import("./pages/TelemetryExplorer"));
const DetectionCentre = lazy(() => import("./pages/DetectionCentre"));
const Monitor = lazy(() => import("./pages/Monitor"));
const Investigation = lazy(() => import("./pages/Investigation"));
const DatasetExplorer = lazy(() => import("./pages/DatasetExplorer"));
const ModelLab = lazy(() => import("./pages/ModelLab"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const Experiments = lazy(() => import("./pages/Experiments"));
const ExperimentDetail = lazy(() => import("./pages/ExperimentDetail"));
const Docs = lazy(() => import("./pages/Docs"));

class Boundary extends Component<{ children: ReactNode }, { error: unknown }> {
  state = { error: null as unknown };
  static getDerivedStateFromError(error: unknown) {
    return { error };
  }
  render() {
    return this.state.error ? <ErrorState title="This page failed to render" error={this.state.error} /> : this.props.children;
  }
}

export default function App() {
  return (
    <Shell>
      <Boundary>
        <Suspense fallback={<Loading height={400} />}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/telemetry" element={<TelemetryExplorer />} />
            <Route path="/detect" element={<DetectionCentre />} />
            <Route path="/monitor" element={<Monitor />} />
            <Route path="/investigate" element={<Investigation />} />
            <Route path="/investigate/:id" element={<Investigation />} />
            <Route path="/dataset" element={<DatasetExplorer />} />
            <Route path="/lab" element={<ModelLab />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/experiments/:id" element={<ExperimentDetail />} />
            <Route path="/docs" element={<Docs />} />
            <Route path="*" element={<ErrorState title="Page not found" error="There is no page at this address." />} />
          </Routes>
        </Suspense>
      </Boundary>
    </Shell>
  );
}
