"""
compare_runtimes_betw_models.py
--------------------
Usage:
    python compare_runtimes.py <csv_file_1> <csv_file_2> [output_file.csv]

Compares per-query runtimes between two model result CSVs.
- Model prefix "HM" -> "Learned Model"
- Model prefix "DB" -> "SQL Server"
- Runtime Sum of -1 -> Timeout
- Runtime Sum of -2 -> Bad Plan

Output CSV columns:
    Query, Model 1 Runtime (ms), Model 2 Runtime (ms), Faster Model, Difference

Two summaries at the top of the output CSV:
  1. All comparable queries (at least one model ran successfully)
  2. Head-to-head only (both models ran successfully)
  
Average runtime difference (mean & median) included in both summaries 
(calculated only from queries where both models produced valid runtimes).
"""

import csv
import sys
import os
from collections import defaultdict


# ── helpers ───────────────────────────────────────────────────────────────────


MODEL_LABELS = {"HM": "Learned Model", "DB": "SQL Server"}


def label(prefix: str) -> str:
    return MODEL_LABELS.get(prefix.strip().upper(), prefix.strip())


def classify_time(val: float) -> str:
    """Return a human-readable status for special sentinel values, or None if valid."""
    if val == -1:
        return "Timeout"
    if val == -2:
        return "Bad Plan"
    return None


def parse_csv(filepath: str) -> dict:
    """
    Returns a dict: { job_name -> {"prefix": str, "time": float, "status": str|None} }
    'status' is None for valid runtimes, or "Timeout" / "Bad Plan" for failures.
    """
    records = {}
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            prefix = row.get("", "")
            job = row.get("Job", "").strip()
            try:
                runtime = float(row.get("Time", -1))
            except ValueError:
                runtime = -1.0
            status = classify_time(runtime)
            records[job] = {"prefix": prefix, "time": runtime, "status": status}
    return records


def runtime_display(rec: dict) -> str:
    if rec["status"]:
        return rec["status"]
    return f"{rec['time']:,.0f}"


# ── main ──────────────────────────────────────────────────────────────────────


def compare(file1: str, file2: str, outfile: str):
    data1 = parse_csv(file1)
    data2 = parse_csv(file2)

    prefix1 = next(iter(data1.values()))["prefix"] if data1 else ""
    prefix2 = next(iter(data2.values()))["prefix"] if data2 else ""
    model1 = label(prefix1) if prefix1 else os.path.splitext(os.path.basename(file1))[0]
    model2 = label(prefix2) if prefix2 else os.path.splitext(os.path.basename(file2))[0]

    all_jobs = sorted(set(data1) | set(data2))
    total_queries = len(all_jobs)

    # ── per-query comparison ──────────────────────────────────────────────────
    results      = []
    wins         = defaultdict(int)  # wins where at least one model ran
    wins_both    = defaultdict(int)  # wins where BOTH models ran
    compared     = 0                 # queries where at least one model ran
    both_ran     = 0                 # queries where both models ran successfully
    differences  = []                # list of all actual numeric differences

    for job in all_jobs:
        rec1 = data1.get(job)
        rec2 = data2.get(job)

        rt1_display = runtime_display(rec1) if rec1 else "N/A"
        rt2_display = runtime_display(rec2) if rec2 else "N/A"

        valid1 = rec1 is not None and rec1["status"] is None
        valid2 = rec2 is not None and rec2["status"] is None

        if valid1 and valid2:
            compared += 1
            both_ran += 1
            diff = rec2["time"] - rec1["time"]
            differences.append(diff)

            if diff > 0:
                faster  = model1
                diff_ms = f"{diff:,.0f} ms faster"
                wins[model1]      += 1
                wins_both[model1] += 1
            elif diff < 0:
                faster  = model2
                diff_ms = f"{diff:,.0f} ms slower"
                wins[model2]      += 1
                wins_both[model2] += 1
            else:
                faster  = "Tie"
                diff_ms = "0 ms (tie)"
        elif valid1 and not valid2:
            faster  = model1
            diff_ms = f"{model2} failed ({rec2['status'] if rec2 else 'N/A'})"
            wins[model1] += 1
            compared += 1
        elif valid2 and not valid1:
            faster  = model2
            diff_ms = f"{model1} failed ({rec1['status'] if rec1 else 'N/A'})"
            wins[model2] += 1
            compared += 1
        else:
            faster  = "N/A (both failed or missing)"
            diff_ms = ""

        results.append({
            "Query":                  job,
            f"{model1} Runtime (ms)": rt1_display,
            f"{model2} Runtime (ms)": rt2_display,
            "Faster Model":           faster,
            "Difference":             diff_ms,
        })

    # ── compute tallies ───────────────────────────────────────────────────────
    w1  = wins[model1]
    w2  = wins[model2]
    p1  = (w1 / compared * 100) if compared else 0
    p2  = (w2 / compared * 100) if compared else 0

    if w1 > w2:
        overall_winner, overall_wins, overall_pct = model1, w1, p1
    elif w2 > w1:
        overall_winner, overall_wins, overall_pct = model2, w2, p2
    else:
        overall_winner, overall_wins, overall_pct = "Tie", w1, p1

    wb1 = wins_both[model1]
    wb2 = wins_both[model2]
    pb1 = (wb1 / both_ran * 100) if both_ran else 0
    pb2 = (wb2 / both_ran * 100) if both_ran else 0

    if wb1 > wb2:
        hth_winner, hth_wins, hth_pct = model1, wb1, pb1
    elif wb2 > wb1:
        hth_winner, hth_wins, hth_pct = model2, wb2, pb2
    else:
        hth_winner, hth_wins, hth_pct = "Tie", wb1, pb1

    # Compute mean and median from collected differences
    diff_count = len(differences)
    mean_diff = sum(differences) / diff_count if diff_count else 0
    
    differences_sorted = sorted(differences)
    if diff_count % 2 == 1:
        median_diff = differences_sorted[diff_count // 2]
    else:
        median_diff = (differences_sorted[diff_count // 2 - 1] + differences_sorted[diff_count // 2]) / 2

    # ── build summary rows ────────────────────────────────────────────────────
    summary_lines = [
        ["=== SUMMARY: ALL COMPARABLE QUERIES ==="],
        ["Includes queries where at least one model executed successfully (other may have timed out or bad plan)"],
        [f"Total queries compared: {compared} of {total_queries}"],
        [f"{model1} faster: {w1} queries ({p1:.1f}%)"],
        [f"{model2} faster: {w2} queries ({p2:.1f}%)"],
    ]
    if overall_winner == "Tie":
        summary_lines.append([f"Result: TIE — both models performed equally on {overall_wins} queries"])
    else:
        summary_lines.append([f"Winner: {overall_winner} — faster on {overall_wins} of {compared} comparable queries ({overall_pct:.1f}%)"])

    summary_lines += [
        [],
        ["=== SUMMARY: HEAD-TO-HEAD (BOTH MODELS EXECUTED SUCCESSFULLY) ==="],
        ["Excludes any query where either model timed out or produced a bad plan"],
        [f"Total queries where both executed: {both_ran} of {total_queries}"],
        [f"Mean difference (n={diff_count}): {mean_diff:,.2f} ms"],
        [f"Median difference (n={diff_count}): {median_diff:,.2f} ms"],
        [f"{model1} faster: {wb1} queries ({pb1:.1f}%)"],
        [f"{model2} faster: {wb2} queries ({pb2:.1f}%)"],
    ]
    if hth_winner == "Tie":
        summary_lines.append([f"Result: TIE — both models performed equally on {hth_wins} head-to-head queries"])
    else:
        summary_lines.append([f"Winner: {hth_winner} — faster on {hth_wins} of {both_ran} head-to-head queries ({hth_pct:.1f}%)"])

    summary_lines.append([])  # blank spacer before data table

    # ── write output ──────────────────────────────────────────────────────────
    fieldnames = [
        "Query",
        f"{model1} Runtime (ms)",
        f"{model2} Runtime (ms)",
        "Faster Model",
        "Difference",
    ]

    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for line in summary_lines:
            writer.writerow(line)
        writer.writerow(fieldnames)
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in results:
            dict_writer.writerow(row)

    print(f"Output written to: {outfile}")
    print()
    for line in summary_lines:
        if line:
            print(line[0])


# ── entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_runtimes.py <file1.csv> <file2.csv> [output.csv]")
        sys.exit(1)

    f1  = sys.argv[1]
    f2  = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "comparison_result.csv"

    for path in (f1, f2):
        if not os.path.exists(path):
            print(f"Error: file not found: {path}")
            sys.exit(1)

    compare(f1, f2, out)