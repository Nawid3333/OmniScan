<script lang="ts">
  import { getJob, postRun } from "./api";
  import type { Job } from "./api";
  import { POLL_INTERVAL_MS, phaseOf } from "./run";
  import type { RunPhase, RunState } from "./run";

  let {
    series,
    chapter,
    through,
    label,
    onDone,
  }: {
    series: string;
    chapter: string;
    through: string;
    label?: string;
    onDone: () => void;
  } = $props();

  let runState = $state<RunState>({ phase: "idle", error: null });
  let stages = $state<string[]>([]);

  let buttonText = $derived(label ?? `Run through ${through}`);

  // Polling bookkeeping. `epoch` invalidates in-flight async work (a click's POST or a poll) when
  // the series/chapter changes underneath it, so no timer or state write survives a chapter switch.
  let timer: ReturnType<typeof setInterval> | null = null;
  let epoch = 0;

  function stopPolling(): void {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  // A series/chapter change (or unmount) resets the button and tears down any in-flight poll.
  $effect(() => {
    const s = series;
    const c = chapter;
    epoch += 1;
    stopPolling();
    runState = { phase: "idle", error: null };
    stages = [];
    return stopPolling;
  });

  async function start(): Promise<void> {
    const myEpoch = ++epoch;
    stopPolling();
    try {
      const result = await postRun(series, chapter, through);
      if (myEpoch !== epoch) return;
      stages = result.stages; // shown immediately, before the first poll resolves
      runState = { phase: "queued", error: null };
      timer = setInterval(() => void poll(result.job_id, myEpoch), POLL_INTERVAL_MS);
    } catch (e) {
      if (myEpoch !== epoch) return;
      runState = { phase: "failed", error: e instanceof Error ? e.message : String(e) };
    }
  }

  async function poll(jobId: number, myEpoch: number): Promise<void> {
    if (myEpoch !== epoch) {
      stopPolling();
      return;
    }
    let job: Job;
    try {
      job = await getJob(jobId);
    } catch {
      return; // transient fetch failure: keep the current phase, the next tick retries
    }
    if (myEpoch !== epoch) return;
    stages = job.stages;
    const phase = phaseOf(job.status);
    runState = { phase, error: phase === "failed" ? job.error : null };
    if (phase === "failed" || phase === "done") {
      stopPolling();
      if (phase === "done") onDone();
    }
  }
</script>

{#if runState.phase === "idle" || runState.phase === "failed"}
  {#if runState.phase === "failed" && runState.error !== null}
    <p style="color:#dc2626;">{runState.error}</p>
  {/if}
  <button onclick={() => void start()}>{runState.phase === "failed" ? "Retry" : buttonText}</button>
{:else}
  <span>{runState.phase}: {stages.join(", ")}…</span>
{/if}