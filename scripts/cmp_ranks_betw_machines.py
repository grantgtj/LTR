import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import torch
import matplotlib.pyplot as plt

# ── MODIFICATION NOTE ────────────────────────────────────────────────────────
# Original script (cmp_ranks_betw_cost_runtime.py) compared cost-based rank
# vs runtime-based rank for plans run on the SAME machine.
#
# This version instead compares runtime-based ranks from TWO DIFFERENT machines
# (e.g. your Mac and the VM), using the same enumerated random plans
# (0.txt … N.txt) executed on the same set of queries (Jobs).
#
# Changes made to the original:
#   1. Two CSV paths are defined (one per machine) instead of one.
#   2. Both DataFrames are loaded, filtered for valid runs (Time != -1),
#      and inner-joined on [Job, Plan] so only plans that ran successfully
#      on BOTH machines are compared.
#   3. The merged DataFrame has columns Time_A and Time_B; these replace
#      the original runtime_labels / cost_labels respectively.
#   4. Job grouping now uses the "Job" column (your CSV) instead of "Job_nr".
#   5. Minor bug-fix in plot_ndcg: parameter names used instead of undefined
#      local variables mean_ndcg / median_ndcg.
#
# Everything else — calculate_linear_raw_scores, the topk loop, the NDCG
# computation, the statistics, and the plot — is kept exactly as in the
# original.
# ─────────────────────────────────────────────────────────────────────────────

def calculate_linear_raw_scores(scores):
    print('runtimes: ', scores)
    scores_sorted = list(np.sort(scores)[::-1])
    # print('scores_sorted: ', scores_sorted)
    score_sorted_indices = []
    for idx, i in enumerate(scores):
        index = scores_sorted.index(i)
        score_sorted_indices.append(index)
        scores_sorted[index] = "abcde"
    print('scores: ', score_sorted_indices)
    return score_sorted_indices


# ── MODIFICATION 1: define one path per machine instead of one combined path ─
machine_A_path = "./TestResults/2_TPCH_d/GrantMacbook/dataset_tpch_workload_tpch-d_enum_HM_rank_None_iter_RandomPlans_runtime.csv"   # e.g. your Mac results
machine_B_path = "./TestResults/2_TPCH_d/VM/dataset_tpch_workload_tpch-d_enum_HM_rank_None_iter_RandomPlans_runtime.csv"   # e.g. the VM results

# ── MODIFICATION 2: load and merge the two machine CSVs ──────────────────────
df_A = pd.read_csv(machine_A_path)
df_B = pd.read_csv(machine_B_path)

if "Unnamed: 0" in df_A.columns:
    df_A = df_A.rename(columns={"Unnamed: 0": "Machine"})
if "Unnamed: 0" in df_B.columns:
    df_B = df_B.rename(columns={"Unnamed: 0": "Machine"})

df_A_valid = df_A[df_A["Time"] != -1][["Job", "Plan", "Time"]]
df_B_valid = df_B[df_B["Time"] != -1][["Job", "Plan", "Time"]]

df_merged = pd.merge(
    df_A_valid, df_B_valid,
    on=["Job", "Plan"],
    suffixes=("_A", "_B")
)

df_merged = df_merged.groupby("Job").filter(
    lambda g: not ((g["Time_A"] == -2).any() or (g["Time_B"] == -2).any())
)

# df_used mirrors the role of df_used in the original: the ready-to-use table
df_used = df_merged.dropna(subset=["Time_A", "Time_B"])

# ── MODIFICATION 3: use "Job" instead of "Job_nr" ────────────────────────────
Job_nr_list = df_used["Job"].unique().tolist()

ndcg_sk_list = []
topk_list = [3, 6, 10, 15]
mean_ndcg_list = []
median_ndcg_list = []

# Output CSV path — all results will be written here
output_csv_path = "./ndcg_results.csv"

# Collectors for the two output tables
per_query_rows = []   # one row per (topk, job)
summary_rows   = []   # one row per topk

for topk in topk_list:
    ndcg_sk_list = []   # reset per topk (matches original behaviour)

    for job_nr in Job_nr_list:
        job_df = df_used[df_used["Job"] == job_nr]

        # ── MODIFICATION 4: use Time_A / Time_B instead of Cost / Time ───────
        # Time_A (Machine A) acts as the "true" relevance reference.
        # Time_B (Machine B) acts as the "predicted" scores to be evaluated.
        # This is a direct analogue of the original's runtime_labels / cost_labels.
        runtime_labels_A = job_df["Time_A"].values   # reference machine
        runtime_labels_B = job_df["Time_B"].values   # machine to evaluate

        runtime_labels_A = calculate_linear_raw_scores(runtime_labels_A)
        print('####')
        runtime_labels_B = calculate_linear_raw_scores(runtime_labels_B)

        print('job_nr', job_nr)
        print('machine_A_labels: ', runtime_labels_A)
        print('machine_B_labels: ', runtime_labels_B)

        # Skip jobs that don't have enough common valid plans for this k.
        # (mirrors the original's implicit assumption that every job has
        # enough plans; here we guard explicitly since plans that fail on
        # one machine are dropped, potentially leaving fewer than topk)
        if len(runtime_labels_A) < 2:
            print(f'{job_nr}: skipped — fewer than 2 common valid plans')
            per_query_rows.append({
                "topk":            topk,
                "job":             job_nr,
                "n_common_plans":  len(runtime_labels_A),
                "runtimes_A":      str(job_df["Time_A"].values.tolist()),
                "runtimes_B":      str(job_df["Time_B"].values.tolist()),
                "rank_labels_A":   str(runtime_labels_A),
                "rank_labels_B":   str(runtime_labels_B),
                "ndcg":            None,
                "skipped":         True,
            })
            continue

        ndcg_sk = ndcg_score(
            np.array([runtime_labels_A]),
            np.array([runtime_labels_B]),
            k=topk
        )
        print(f'{job_nr}, ndcg_sk', ndcg_sk)

        ndcg_sk_list.append(ndcg_sk)

        # Collect this row for CSV output
        per_query_rows.append({
            "topk":           topk,
            "job":            job_nr,
            "n_common_plans": len(runtime_labels_A),
            "runtimes_A":     str(job_df["Time_A"].values.tolist()),
            "runtimes_B":     str(job_df["Time_B"].values.tolist()),
            "rank_labels_A":  str(runtime_labels_A),
            "rank_labels_B":  str(runtime_labels_B),
            "ndcg":           ndcg_sk,
            "skipped":        False,
        })

    print("the number of jobs: ", len(Job_nr_list))
    print("mean ndcg over all queries: ", np.mean(ndcg_sk_list))

    ndcg_df = pd.DataFrame(ndcg_sk_list)
    ndcg_df.columns = ['NDCG']
    print(ndcg_df.describe())

    mean_ndcg  = ndcg_df.describe().loc["mean", 'NDCG']
    median_ndcg = ndcg_df.describe().loc["50%",  'NDCG']

    mean_ndcg_list.append(mean_ndcg)
    median_ndcg_list.append(median_ndcg)

    # Collect summary row for this topk
    desc = ndcg_df.describe()
    summary_rows.append({
        "topk":               topk,
        "n_jobs":             len(Job_nr_list),
        "n_scored":           len(ndcg_sk_list),
        "mean_ndcg":          desc.loc["mean",  "NDCG"],
        "median_ndcg":        desc.loc["50%",   "NDCG"],
        "std_ndcg":           desc.loc["std",   "NDCG"],
        "min_ndcg":           desc.loc["min",   "NDCG"],
        "max_ndcg":           desc.loc["max",   "NDCG"],
    })

print('mean',   mean_ndcg_list)
print('median', median_ndcg_list)

# ── Write all results to a single CSV ────────────────────────────────────────
# The file has two sections separated by a blank row:
#   Section 1 — per-query: one row per (topk, job)
#   Section 2 — summary:   one row per topk (mean/median/std/min/max NDCG)
per_query_df = pd.DataFrame(per_query_rows)
summary_df   = pd.DataFrame(summary_rows)

with open(output_csv_path, "w") as f:
    f.write("=== per_query ===\n")
    per_query_df.to_csv(f, index=False)
    f.write("\n=== summary ===\n")
    summary_df.to_csv(f, index=False)

print(f'\nResults written to: {output_csv_path}')


# ── MODIFICATION 5 (bug-fix): use the function's parameter names ──────────────
# Original used bare `mean_ndcg` / `median_ndcg` which are only defined in the
# outer scope after the loop; the function parameters were shadowed/unused.
def plot_ndcg(topk_list, mean_ndcg_list, median_ndcg_list, db="tpch"):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    barWidth = 0.25
    br1 = np.arange(len(topk_list))
    br2 = [x + barWidth for x in br1]

    ax.bar(br1, mean_ndcg_list,   color='r', width=barWidth, edgecolor='k', label='Mean')
    ax.bar(br2, median_ndcg_list, color='g', width=barWidth, edgecolor='k', label='Median')

    plt.xlabel(f'K values ({db})', fontsize=20)
    plt.ylabel('NDCG@K', fontsize=20)
    plt.ylim(-0.01, 1.1)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=20)
    plt.xticks(
        [r + 0.5 * barWidth for r in range(len(topk_list))],
        [topk_list[r] for r in range(len(topk_list))],
        fontsize=20
    )
    plt.legend(ncol=2, fontsize=10)
    plt.tight_layout()
    plt.show()


plot_ndcg(topk_list, mean_ndcg_list, median_ndcg_list, db="JOB")