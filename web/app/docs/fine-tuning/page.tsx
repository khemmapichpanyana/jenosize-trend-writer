import { Prose, Callout } from "@/components/docs/prose";
import { DocsPager } from "@/components/docs/docs-pager";

export const metadata = { title: "Fine-tuning" };

export default function FineTuningPage() {
  return (
    <>
    <Prose>
      <h1>Model selection &amp; fine-tuning</h1>

      <h2>Why Qwen3-4B-Instruct-2507</h2>
      <p>
        A 4B-parameter instruction model is small enough to serve on a single Modal GPU (scale-to-zero, one
        L4) while retaining solid English business-writing ability, so there is no need to reach for the
        largest model available. The <em>2507</em> release is the &ldquo;non-thinking&rdquo;
        variant: it doesn&rsquo;t emit <code>&lt;think&gt;</code> reasoning blocks, which would otherwise leak into
        training data and into the final article text.
      </p>
      <p>
        On top of it: <strong>QLoRA</strong> (4-bit base weights, LoRA adapters on all attention and MLP
        projection layers) rather than full fine-tuning. Style transfer — tone, structure, the article&rsquo;s
        output contract — doesn&rsquo;t need every weight in the model updated, and QLoRA keeps a training run
        inside the memory of one 24&nbsp;GB GPU and inside minutes rather than hours.
      </p>

      <h2>Training configuration</h2>
      <table>
        <thead><tr><th>Setting</th><th>Value</th><th>Why</th></tr></thead>
        <tbody>
          <tr><td>Base model</td><td><code>Qwen/Qwen3-4B-Instruct-2507</code></td><td>Small, strong, no thinking-block leakage</td></tr>
          <tr><td>Method</td><td>QLoRA — 4-bit base, LoRA on attention + MLP projections</td><td>Style transfer fits on one L4 GPU</td></tr>
          <tr><td>LoRA rank</td><td>16 (serving supports up to 64)</td><td>Standard QLoRA starting point</td></tr>
          <tr><td>Loss</td><td>Assistant turn only (Qwen ChatML markers)</td><td>Otherwise the model learns to write <em>briefs</em>, not articles</td></tr>
        </tbody>
      </table>

      <h2>Dataset</h2>
      <p>
        Built from Jenosize Ideas&rsquo; own published articles (see <a href="/docs/how-it-works">How it
        works</a> for the discover → clean → label → build pipeline), not a synthetic or third-party corpus:
      </p>
      <table>
        <thead><tr><th>Stage</th><th>Count</th></tr></thead>
        <tbody>
          <tr><td>URLs discovered (sitemap)</td><td>174</td></tr>
          <tr><td>Fetched</td><td>170</td></tr>
          <tr><td>Cleaned &amp; reverse-labelled</td><td>159</td></tr>
          <tr><td>v2 quality-filtered corpus</td><td>105 (92 train / 13 held-out eval)</td></tr>
        </tbody>
      </table>
      <p>
        The v2 quality filter excluded 54 candidates before publishing the split: 9 failed title-length, 12
        failed meta-description length, and 43 had fewer than three H2 sections. Splitting is deterministic
        (SHA-256 of the source URL, seed 13, 10% eval fraction) and reproducible — re-running the pipeline
        against the same corpus produces the same split. Dataset fingerprint: <code>69f0b4147fd8&hellip;</code>.
        Accepted examples run 548–1,603 words (median 839); the largest categories are Transformation &amp;
        Technology, Real-time Marketing, and Understand People &amp; Consumer.
      </p>

      <Callout tone="warn" title="Known dataset limitation">
        The corpus is English-only, single-publisher, and its briefs are <em>inferred</em> from finished
        articles rather than real editorial instructions — there are no source-document-grounded training
        examples, so retrieval/search grounding is carried by the base model and the inference prompt, not
        learned by the adapter. See <a href="/docs/results">Results &amp; evidence</a> for what this means for
        quality.
      </Callout>

      <h2>Two trained versions</h2>
      <table>
        <thead><tr><th></th><th>v1</th><th>v2 (active)</th></tr></thead>
        <tbody>
          <tr><td>Training examples</td><td>146</td><td>92</td></tr>
          <tr><td>Epochs</td><td>3</td><td>1</td></tr>
          <tr><td>Optimizer steps</td><td>57</td><td>12</td></tr>
          <tr><td>Learning rate</td><td>2e-4</td><td>1e-4</td></tr>
          <tr><td>LoRA rank</td><td>16</td><td>16</td></tr>
          <tr><td>Final train loss</td><td>1.8873</td><td>2.2831</td></tr>
        </tbody>
      </table>
      <p>
        The two losses are not directly comparable — the corpus, quality filter, epoch count and learning
        rate all changed between v1 and v2. v1 performed materially worse in review, so v2 (a smaller,
        quality-filtered corpus) is the deployed, served-as-<code>jeno-lora</code> alias.
      </p>

      <h2>How the adapter is served</h2>
      <p>
        Each trained version lands in its own directory on a persistent Modal volume
        (<code>/models/jeno-lora-{"{version}"}</code>) so a new version never overwrites an old one. The vLLM
        server registers <em>every</em> trained version under its own name (<code>jeno-lora-v1</code>,
        <code> jeno-lora-v2</code>, &hellip;) plus a movable <code>jeno-lora</code> alias for whichever one is
        currently active — activation just repoints the alias for the next cold start, no redeploy needed.
        vLLM is configured to allow LoRA ranks up to 64, since its own default of 16 would refuse a
        higher-rank adapter later.
      </p>
    </Prose>
    <DocsPager current="/docs/fine-tuning" />
    </>
  );
}
