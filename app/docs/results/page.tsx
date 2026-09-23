import { Prose, Callout } from "@/components/docs/prose";
import { DocsPager } from "@/components/docs/docs-pager";

export const metadata = { title: "Results & evidence" };

export default function ResultsPage() {
  return (
    <>
    <Prose>
      <h1>Results &amp; evidence</h1>
      <p>
        Evaluation replays the same 13 held-out briefs (never seen during training) against the base model
        and the fine-tuned <code>jeno-lora-v2</code> adapter, served by the same vLLM instance so the adapter
        is the only variable that differs.
      </p>

      <h2>Evaluation methodology</h2>
      <ul>
        <li><strong>Deterministic quality gate</strong> — the same four checks the product API applies to every
          generated article: word count, H2 section count, SEO keyword coverage, and title length.</li>
        <li><strong>Blind pairwise judge</strong> (optional) — a separate labelling LLM compares the two
          outputs for a brief without knowing which is which; presentation order is randomised to cancel
          position bias.</li>
      </ul>

      <h2>13-brief evaluation results</h2>
      <table>
        <thead><tr><th>Measure</th><th>Base</th><th>Fine-tuned v2</th></tr></thead>
        <tbody>
          <tr><td>Deterministic quality-gate pass rate</td><td>69.2% (9/13)</td><td><strong>76.9% (10/13)</strong></td></tr>
          <tr><td>Mean SEO keyword coverage</td><td><strong>95.8%</strong></td><td>92.1%</td></tr>
          <tr><td>Mean article length</td><td>705 words</td><td>744 words</td></tr>
          <tr><td>Mean H2 sections</td><td>4.38</td><td>4.92</td></tr>
        </tbody>
      </table>
      <p>
        The blind judge recorded <strong>3 fine-tuned wins and 2 base wins</strong>, but only 5 of the 13
        comparisons produced a parseable verdict, so this is not a full 13-pair result.
      </p>

      <Callout tone="evidence" title="Honest read of these numbers">
        The fine-tuned model wins the quality gate and produces slightly longer, better-structured articles,
        but loses a little SEO keyword coverage, and the judge sample is too small (5 of 13) to call. This is
        evidence that v2 is a reasonable demo checkpoint — it is <em>not</em> evidence that fine-tuning
        universally improves quality, and it should not be presented as such. v1 (146 examples, 3 epochs)
        performed materially worse than base in review, which is itself useful evidence: more training
        data/epochs did not straightforwardly mean a better adapter here — the quality-filtered, smaller v2
        corpus did better.
      </Callout>

      <h2>Weights &amp; where they live</h2>
      <ul>
        <li><strong>Base weights</strong> — <code>Qwen/Qwen3-4B-Instruct-2507</code>, downloaded from Hugging Face
          into the Modal image&rsquo;s cache volume, unmodified.</li>
        <li><strong>Adapter weights</strong> — LoRA-only (rank 16), a few tens of MB, not the full 4B parameters.
          Trained with Unsloth, stored per-version on a persistent Modal Volume at
          <code> /models/jeno-lora-{"{version}"}</code>, never overwritten.</li>
        <li><strong>Registry</strong> — <code>GET /v1/adapters</code> lists every trained version with its
          training loss, example count, epochs and learning rate (the table above is read straight from
          this endpoint) — see the <strong>Models</strong> tab in this console for the live view.</li>
        <li><strong>Active alias</strong> — vLLM serves every trained version under its own name plus a
          movable <code>jeno-lora</code> alias for the one currently in front of the product; switching it is
          a metadata change, not a retrain.</li>
      </ul>

      <h2>Product-level observation</h2>
      <p>
        A warm vLLM reply is a matter of seconds; a scale-to-zero cold start was observed at roughly 2–3
        minutes end to end, so the console sends a best-effort warmup request on entering Studio and the API
        retries transient <code>502</code>/<code>503</code>/<code>504</code> responses rather than surfacing
        them as failures — cold-start latency is real and should be described as such, not hidden.
      </p>

      <h2>Limitations &amp; next steps</h2>
      <ul>
        <li>The dataset is small, English-only and single-publisher; its briefs are inferred from finished
          articles rather than real editorial instructions.</li>
        <li>Training examples have no source-document grounding — the adapter was never taught to cite, so
          factual grounding relies entirely on retrieval/<code>web_search</code> at inference time, and
          generated claims still need editorial fact-checking before publishing.</li>
        <li>Next steps: human review across more held-out topics (especially non-English), a complete
          13-pair blind adjudication instead of 5, explicit citation/evidence checks on generated claims, and
          a persistent warm serving tier only if real usage volume justifies the added cost.</li>
      </ul>
    </Prose>
    <DocsPager current="/docs/results" />
    </>
  );
}
