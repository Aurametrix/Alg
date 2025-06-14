# ── 1.  Load: one row per evaluation ─────────────────────────────────────
# columns = ["model", "benchmark", "date", "score"]  # score 0-100
df = pd.read_csv("llm_benchmark_scores.csv", parse_dates=["date"])

# ── 2.  Cumulative ceiling per benchmark over time  ─────────────────────
df = df.sort_values("date")
df["ceiling_t"] = (
    df.groupby("benchmark")["score"]
      .expanding()
      .max()
      .reset_index(level=0, drop=True)
)

# ── 3.  Global ceiling  ─────────────────────────────────────────────────
global_ceiling = (
    df.groupby("benchmark")["score"]
      .max()
      .rename("ceiling_max")
)
