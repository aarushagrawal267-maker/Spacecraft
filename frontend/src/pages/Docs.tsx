import { useDatasetStatus, useMethods } from "../api/hooks";
import { PageHeader } from "../components/ui";
import { fmtInt } from "../lib/format";

const SECTIONS = [
  ["background", "Project background"],
  ["anomalies", "Spacecraft anomaly detection"],
  ["lstm", "How the LSTM works"],
  ["dataset", "Dataset documentation"],
  ["architecture", "Model architecture"],
  ["thresholding", "Scoring & dynamic thresholding"],
  ["evaluation", "Evaluation methodology"],
  ["limitations", "Limitations"],
  ["reproducibility", "Reproducibility"],
  ["references", "References"],
] as const;

export default function Docs() {
  const ds = useDatasetStatus();
  const m = useMethods();
  const s = ds.data?.snapshot?.summary;
  return (
    <div>
      <PageHeader eyebrow="Research & documentation" title="Method, data and limitations">
        Heliograph is an independent educational and research implementation inspired by published work from NASA's Jet Propulsion
        Laboratory. It is not affiliated with, commissioned by, or used operationally by NASA or JPL.
      </PageHeader>
      <div className="grid gap-10 lg:grid-cols-[200px_1fr]">
        <nav className="sticky top-4 hidden self-start lg:block" aria-label="Sections">
          {SECTIONS.map(([id, label]) => (
            <a key={id} href={`#${id}`} className="block border-l border-hairline py-1.5 pl-3 text-sm text-muted hover:border-accent hover:text-ink">
              {label}
            </a>
          ))}
        </nav>
        <article className="prose-doc panel max-w-3xl px-8 py-6">
          <h2 id="background">Project background</h2>
          <p>
            Spacecraft continuously stream telemetry — measurements from on-board sensors and subsystems. Operations teams watch thousands of
            such channels, and most of the time the values follow recognisable patterns. Hundman et al. (KDD 2018) showed that per-channel
            LSTM forecasting combined with an unsupervised, nonparametric thresholding of forecast errors can flag a large share of
            expert-labelled anomalies in telemetry from NASA's SMAP satellite and the Curiosity rover while keeping false alarms low.
          </p>
          <p>
            This application re-implements that approach from the published description and code, adds comparison baselines and an
            experimental LSTM autoencoder, and wraps everything in tools for exploration, training, evaluation and inspection. It does not
            claim to reproduce the paper's numbers.
          </p>

          <h2 id="anomalies">Spacecraft anomaly detection</h2>
          <p>An <strong>anomaly</strong> here is an unusual pattern that warrants investigation. It is not automatically a malfunction. The application distinguishes four things:</p>
          <ul>
            <li><strong>Model-detected anomaly</strong> — an interval where a model's smoothed error exceeded its threshold (violet).</li>
            <li><strong>Expert-labelled anomaly</strong> — a reference interval from the benchmark's <code>labeled_anomalies.csv</code> (amber, hatched).</li>
            <li><strong>Normal telemetry</strong> — neither flagged nor labelled. Unlabelled is not proof of normal.</li>
            <li><strong>Uncertain or unexplained behaviour</strong> — a model detection with no label overlap. It counts as a false positive in the benchmark, but may still be interesting.</li>
          </ul>
          <p>
            The benchmark labels two classes: <em>point</em> anomalies (values far outside the normal range) and <em>contextual</em> anomalies
            (values that are only unusual given their surrounding sequence). Spikes, drift, level shifts and unusual fluctuations all occur.
          </p>

          <h2 id="lstm">How the LSTM works</h2>
          <p>
            A Long Short-Term Memory network (Hochreiter &amp; Schmidhuber, 1997) is a recurrent neural network whose cells carry a memory
            state through time, controlled by learned input, forget and output gates. This lets it learn temporal dependencies over long
            windows without the vanishing-gradient problems of plain RNNs.
          </p>
          <p>
            For each telemetry channel we train a separate LSTM on the channel's training split. The model sees the previous{" "}
            <code>window</code> timesteps of every input feature (the telemetry value and the one-hot command inputs) and predicts the next{" "}
            <code>horizon</code> telemetry values. It is trained with mean-squared error and early stopping on a chronological validation split.
            On the test stream, the prediction for time <em>t</em> comes from the window ending at <em>t−1</em>; the absolute difference to the
            actual value is the forecast error. Where the model's learned notion of normal behaviour fails, errors grow.
          </p>

          <h2 id="dataset">Dataset documentation</h2>
          <p>
            The benchmark accompanies the KDD 2018 paper and the Telemanom repository. It contains {s ? fmtInt(s.unique_channels) : "81"} unique
            telemetry channels ({s ? `${s.by_spacecraft.SMAP?.channels} SMAP, ${s.by_spacecraft.MSL?.channels} MSL` : "54 SMAP, 27 MSL"}).
            The label file has {s?.label_rows ?? 82} rows: channel <code>P-2</code> appears twice with overlapping intervals, which is why
            documentation usually quotes 82 channels (55 SMAP). Each channel has:
          </p>
          <ul>
            <li><code>train/&lt;chan&gt;.npy</code> — training sequence, shape (timesteps, features); used to fit models.</li>
            <li><code>test/&lt;chan&gt;.npy</code> — test sequence; anomalies are labelled here.</li>
            <li>A row in <code>labeled_anomalies.csv</code>: <code>chan_id</code>, <code>spacecraft</code>, <code>anomaly_sequences</code> (inclusive start/end index pairs into the test sequence), <code>class</code> (point/contextual per sequence) and <code>num_values</code> (test length).</li>
          </ul>
          <p>
            Column 0 is the telemetry value; the other columns are one-hot encoded command information ({s ? `${s.by_spacecraft.SMAP?.features.join(", ")} features for SMAP, ${s.by_spacecraft.MSL?.features.join(", ")} for MSL` : "25 features for SMAP, 55 for MSL"}).
            Channel identifiers are anonymised: only the first letter is documented as a channel type (e.g. P = power, R = radiation). Time is
            anonymised — there are no timestamps or engineering units.
          </p>
          <p>
            The Telemanom README states that values were scaled to (−1, 1) using the test set's minimum and maximum. In the files obtained here,
            several channels nevertheless exceed that range (up to 258 on <code>M-6</code>), and 14 channels have a constant training value.
            Both observations are reported by the validator and shown in the Dataset Explorer.
          </p>
          <p>
            <strong>Acquisition.</strong> The original S3 download link no longer works. The pipeline tries the Kaggle dataset named in the
            Telemanom README, then a pinned GitHub copy of the original files (TranAD repository), then a manually supplied <code>data.zip</code>.
            Labels always come from the official Telemanom repository at a pinned commit, verified by SHA-256. Every telemetry file is checked
            against the label file (channel inventory, test length = <code>num_values</code>, shapes, finite values, binary command columns), and
            a manifest of SHA-256 checksums is written. The original files are kept read-only.
          </p>

          <h2 id="architecture">Model architecture</h2>
          <table>
            <thead><tr><th>Method</th><th>Purpose</th></tr></thead>
            <tbody>
              {(m.data?.methods ?? []).map((x) => (
                <tr key={x.key}><td>{x.label}</td><td>{x.purpose}</td></tr>
              ))}
            </tbody>
          </table>
          <p>
            <strong>LSTM forecaster</strong> (Telemanom defaults, "paper" profile): window 250, horizon 10, two LSTM layers of 80 units, dropout
            0.3, Adam, batch 64, up to 35 epochs, early stopping (patience 10, min Δ 0.0003), 20 % chronological validation. The "standard"
            profile (64 units, ≤ 20 epochs, patience 5) was used for the reference run to keep CPU time reasonable; "quick" (window 100, 48 units,
            ≤ 8 epochs) is the laptop default. The best validation weights are restored.
          </p>
          <p>
            <strong>LSTM autoencoder</strong>: an encoder LSTM compresses a window (default 100 steps, all features) into a 16-dimensional latent
            vector; a decoder LSTM reconstructs the telemetry column of the window. The score at time <em>t</em> is the mean absolute
            reconstruction error of the window ending at <em>t</em>, which keeps it causal.
          </p>
          <p>
            <strong>Random Forest</strong> was deliberately not included: the training split has no labels, so a supervised classifier could
            only be trained on test labels, which would be leakage.
          </p>

          <h2 id="thresholding">Scoring &amp; dynamic thresholding</h2>
          <ol>
            <li>Raw scores (absolute forecast/reconstruction errors or statistical scores) are taken on test indices ≥ 250 for every method.</li>
            <li>They are smoothed with a causal EWMA, span = batch × window × 5 % = 105 points (identical to <code>pandas.ewm(adjust=True)</code>).</li>
            <li>The stream is processed in windows of 30 batches × 70 points. In each window, ε = μ + zσ is chosen from z ∈ [2.5, 12) to maximise (Δμ/μ + Δσ/σ) / (sequences² + flagged points), with at most 5 sequences and fewer than 50 % of points flagged.</li>
            <li>The same search runs on errors reflected about the window mean, flagging unusually low errors.</li>
            <li>Flags must also exceed 5 % of the window's telemetry inter-percentile range and the smoothed error must exceed 0.05 (scale guard). Flags are buffered by 100 points.</li>
            <li>Pruning: flagged sequences whose peak is not at least p = 13 % above the next-highest peak (ending with the highest non-flagged error) are removed. Single points are discarded.</li>
          </ol>
          <p>
            Deviations from Telemanom, all deliberate: no initial errors are ignored (Telemanom skips up to 500 points and special-cases channel
            C-2, whose anomaly starts at index 290); the EWMA warm-up is not overwritten; the final window evaluates every point after the
            previous window's reporting zone (Telemanom's final window can leave up to 69 points unevaluated). An alternative static threshold
            (training-split μ + kσ) is available for comparison.
          </p>
          <p>
            The per-interval <strong>score</strong> is the margin above the threshold divided by (window mean + sd). It is a severity indicator,
            not a calibrated probability, so no "confidence" is displayed.
          </p>

          <h2 id="evaluation">Evaluation methodology</h2>
          <ul>
            <li><strong>Event-wise</strong> (paper protocol): a labelled sequence overlapped by ≥ 1 detection is a TP, otherwise a FN; a detection overlapping no label is a FP. Precision, recall, F1 and F0.5 are reported. True negatives are undefined, so false alarms are given per 10,000 evaluated points instead of a rate.</li>
            <li><strong>Point-wise</strong>: every evaluated timestep is a sample — TP/FP/FN/TN, precision, recall, F1, TPR, FPR. No point adjustment, which is known to inflate results (Kim et al., 2022).</li>
            <li><strong>Interval quality</strong>: detection delay (first detected index inside the label minus label start), IoU and coverage per detected label; recall per anomaly class.</li>
            <li><strong>Threshold-free curves</strong>: precision–recall and ROC over the smoothed score, pooled across channels after standardising each channel's score by its own mean and sd (label-free).</li>
            <li>Results are reported overall, per spacecraft and per channel, with training and inference times. A random-score control is always evaluated alongside.</li>
          </ul>
          <p><strong>Leakage controls</strong>: the original train/test separation is preserved; models never see test data or labels during training (an automated test makes label access during fitting fail); validation is the last 20 % of the training split; thresholds use no labels; another automated test changes the labels and checks that detections stay identical.</p>

          <h2 id="limitations">Limitations</h2>
          <ul>
            <li>The telemetry is anonymised and preprocessed; the stated test-set-based scaling is a form of look-ahead built into the benchmark that cannot be undone here.</li>
            <li>Labels are intervals from incident reports. About 13 % of SMAP and 12 % of MSL evaluated points are labelled — dense for an anomaly benchmark. Unlabelled regions are not guaranteed normal, and some labels are long, which affects point-wise metrics.</li>
            <li>The benchmark has been criticised for triviality and possible mislabelling (Wu &amp; Keogh); good scores should be read with that in mind.</li>
            <li>Models are per-channel and univariate in their target; cross-channel relationships are ignored. A one-step forecaster can adapt to slow drift and miss it.</li>
            <li>The thresholding contains several hand-set constants inherited from Telemanom; using those published defaults is itself mild indirect tuning to this benchmark.</li>
            <li>Results are not comparable with the paper's figures: preprocessing details, training setup (profile, early stopping, restored weights) and thresholding deviations differ.</li>
            <li>A detection is never a diagnosis. This tool cannot determine whether hardware malfunctioned.</li>
          </ul>

          <h2 id="reproducibility">Reproducibility</h2>
          <p>
            Every experiment stores its method, hyper-parameters, threshold configuration, seed, code commit, dataset checksum, device, worker
            count, platform and library versions, per-epoch losses, per-channel metrics and detections. Model checkpoints and per-timestep arrays
            are kept under <code>artifacts/</code>.
          </p>
          <p>
            The reference experiments shipped in <code>reference/</code> were produced with <code>scripts/run_reference.py</code> on a Linux CPU
            machine and exported with <code>scripts/export_reference.py</code>; they are labelled "Reference" throughout the UI. To recompute them
            locally: <code>backend\.venv\Scripts\python scripts\run_reference.py --workers 2</code>. Exact numbers can differ slightly across
            platforms and PyTorch builds because floating-point operations are not bit-identical.
          </p>

          <h2 id="references">References</h2>
          <ol>
            <li>K. Hundman, V. Constantinou, C. Laporte, I. Colwell, T. Soderstrom. <em>Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.</em> KDD 2018. <a href="https://arxiv.org/abs/1802.04431" target="_blank" rel="noreferrer">arXiv:1802.04431</a> · <a href="https://ntrs.nasa.gov/citations/20210008188" target="_blank" rel="noreferrer">NASA NTRS 20210008188</a></li>
            <li>Telemanom implementation and labels: <a href="https://github.com/khundman/telemanom" target="_blank" rel="noreferrer">github.com/khundman/telemanom</a></li>
            <li>S. Hochreiter, J. Schmidhuber. <em>Long Short-Term Memory.</em> Neural Computation 9(8), 1997.</li>
            <li>P. Malhotra et al. <em>LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.</em> ICML Anomaly Detection Workshop, 2016. arXiv:1607.00148.</li>
            <li>F. T. Liu, K. M. Ting, Z.-H. Zhou. <em>Isolation Forest.</em> IEEE ICDM, 2008.</li>
            <li>S. Kim, K. Choi, H.-S. Choi, B. Lee, S. Yoon. <em>Towards a Rigorous Evaluation of Time-Series Anomaly Detection.</em> AAAI 2022. <a href="https://ojs.aaai.org/index.php/AAAI/article/view/20680" target="_blank" rel="noreferrer">doi:10.1609/aaai.v36i7.20680</a></li>
            <li>R. Wu, E. Keogh. <em>Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress.</em> IEEE TKDE, 2023 (arXiv:2009.13807).</li>
            <li>Data mirror: TranAD repository (S. Tuli et al., VLDB 2022), <a href="https://github.com/imperial-qore/TranAD" target="_blank" rel="noreferrer">github.com/imperial-qore/TranAD</a>, pinned commit, used only as a redistribution of the original Telemanom files.</li>
          </ol>
        </article>
      </div>
    </div>
  );
}
