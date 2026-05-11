from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation
from ltr_db_optimizer.enumeration_algorithm.table_info_5 import TPCHTableInformation5
from ltr_db_optimizer.enumeration_algorithm.table_info_imdb import IMDBTableInformation
from ltr_db_optimizer.enumeration_algorithm.table_info_stats import STATSTableInformation
from pyodbc import ProgrammingError, OperationalError
import os
import pandas as pd
from datetime import datetime
import time
from ltr_db_optimizer.parser import SQLParser
from argparse import ArgumentParser
from ltr_db_optimizer.extra_utils import set_DBMS

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--emd", type=str, default='HM')
    parser.add_argument("--db", type=str, default="tpch")
    parser.add_argument('--tq', type=str, default="tpch-o")
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--do_enum', type=bool, default=False)
    parser.add_argument('--do_runtime', type=bool, default=True)
    parser.add_argument('--iter', type=str, default='None')
    parser.add_argument("--mn", type=str, default='special_50_97_MODEL_LTRankModel1_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False')
    print('start time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    Start_time = datetime.now()
    args = parser.parse_args()
    targeted_enum_method = args.emd
    targeted_database = args.db
    targeted_test_query = args.tq
    targeted_model_name = args.mn
    iter = args.iter
    print('Argumments: ', args)
    cursor = set_DBMS(db=targeted_database)

    ### dataset info.
    if targeted_database == "tpch":
        TableInformation = TPCHTableInformation()
    elif targeted_database == "imdb":
        TableInformation = IMDBTableInformation()
    elif targeted_database == "tpch5":
        TableInformation = TPCHTableInformation5()
    elif targeted_database == "stats":
        TableInformation = STATSTableInformation()

    ### enum method info.
    if targeted_enum_method == "HM":
        from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm import EnumerationAlgorithm
        print("from ltr_db_optimizer.enumeration_algorithm.enumeration_algorithm import EnumerationAlgorithm")

    model = None

    ### test queries info.
    if targeted_test_query == "tpch-o":
        query_folder_name = "tpch_queries"
        query_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/Data/testing_data/{query_folder_name}/"
        testquery = ['1', '3', '5', '6', '10', '12', '14']
    elif targeted_test_query in ["tpch-d", "tpch-l"]:
        query_folder_name = "HM_TPCH_test_queries"
        query_path = f'/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/Data/testing_data/{query_folder_name}/'
        import glob
        file_list = glob.glob(f"{query_path}**/*.txt", recursive=True)
        testquery = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
    elif targeted_test_query == "tpch-s":
        df = pd.read_csv('/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/dataset_tpch_workload_tpch-s_enum_DB_iter_tpchdThre50sNoBitMapCurrentCompaLevelCardEstimateCL140_runtime.csv', header=0)
        df = df[(df['Time'] != 0)]
        print('total number of tpch datafarm queries: ', len(df))
        df = df[(df['Time'] < 50000)]
        print('the number of tpch datafarm queries less than 50 seconds: ', len(df))
        testquery = df['Job'].values.tolist()
        query_folder_name = "HM_TPCH_test_queries"
        query_path = f'/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/Data/testing_data/{query_folder_name}/'
    elif targeted_test_query == "imdb-o":
        query_folder_name = "imdb_queries"
        query_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = [q.split(".")[0] for q in query_file_names if q.endswith(".txt")]
    elif targeted_test_query == "job-o":
        query_folder_name = "job"
        query_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = sorted([q.split(".")[0] for q in query_file_names if q.endswith(".sql")])
    elif targeted_test_query == "stats-o":
        query_folder_name = "stats_queries"
        query_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = [q.split(".")[0] for q in query_file_names if q.endswith(".txt") and q.startswith("qt")]
    elif targeted_test_query == "stats-r":
        query_folder_name = "stats_test2"
        query_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = [q.split(".")[0] for q in query_file_names if q.endswith(".txt") and (q.startswith("qt") or q.startswith("tq"))]
    elif targeted_test_query == "imdb-r":
        query_folder_name = "imdb_test2"
        query_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = sorted([q.split(".")[0] for q in query_file_names if q.endswith(".sql")])
    elif targeted_test_query == "job-t3":
        query_folder_name = "imdb_test3"
        query_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/Data/testing_data/{query_folder_name}/"
        query_file_names = os.listdir(query_path)
        testquery = sorted([q.split(".")[0] for q in query_file_names if q.endswith(".sql")])

    k = args.topk
    enum_plan_time = []

    if args.do_enum:
        for job_index, job in enumerate(testquery):
            if targeted_test_query == "tpch-d" and job in ["Job17v3", "Job1131v3"]:
                continue
            if targeted_test_query == "tpch-l" and job not in ['Job49v1', 'Job44v3', 'Job25v1', 'Job253v4', 'Job253v0',
                                                               'Job185v3', 'Job185v2', 'Job16v0', 'Job161v0', 'Job1325v3',
                                                               'Job1300v2', 'Job128v4', 'Job128v3', 'Job128v0', 'Job1252v1',
                                                               'Job1103v3', 'Job1034v4', 'Job1034v0', 'Job1033v4', 'Job1028v2',
                                                               'Job1021v1', 'Job1017v3', 'Job1017v1', 'Job1011v2']:
                continue
            if targeted_test_query == "tpch-s" and job in ["Job1300v2"]:
                continue
            print('enumerate job: ', job_index, job)
            if os.path.exists(f"{query_path}/{job}.txt"):
                with open(f"{query_path}/{job}.txt", "r") as f:
                    sql_full = f.read()
            else:
                with open(f"{query_path}/{job}.sql", "r") as f:
                    sql_full = f.read()
            sql_full = sql_full.replace("tcph", "tpch")
            path_out = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/enumerated_plans_{targeted_enum_method}_{targeted_model_name}/{query_folder_name}/iter{iter}/{job}/"
            if targeted_test_query in ["job-o", "stats-o"] and os.path.exists(path_out):
                print(f'{job} in {targeted_test_query} alreay enumerated and thus ignored!')
                continue
            sql_dict, alias_dict = SQLParser.from_sql(sql_full, temp_table_info=TableInformation)
            enum = EnumerationAlgorithm(sql_dict, TableInformation, model, sql_full, k, alias_dict=alias_dict, train_wk=[])
            start_time = datetime.now()
            best_plans = enum.find_best_plan()
            enum_end_time = datetime.now()
            enum_delta_time = (enum_end_time - start_time).total_seconds()
            print(f"{job}'s planning time:", enum_delta_time)
            print('the number of returned best plans', len(best_plans))
            os.system(f"mkdir -p {path_out}")
            for idx, plan in enumerate(best_plans):
                xml = enum.to_xml(plan)
                with open(path_out + str(idx) + ".txt", "w") as f:
                    f.write(xml)

    if args.do_runtime:
        print('Enumeration done! Continue to obtain runtime of enumerated plans !!!')
        print('Current time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # ── Determine output CSV paths up front ──────────────────────────────────
        if targeted_enum_method == "DB":
            all_df_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_iter_{iter}_runtime.csv"
            ranking_df_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_iter_{iter}_plan_rankings.csv"
            runtime_desc_file_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_iter_{iter}_runtime_descr.csv"
        else:
            all_df_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_iter_{iter}_runtime.csv"
            ranking_df_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_iter_{iter}_plan_rankings.csv"
            runtime_desc_file_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/dataset_{targeted_database}_workload_{targeted_test_query}_enum_{targeted_enum_method}_rank_{targeted_model_name}_iter_{iter}_runtime_descr.csv"

        # ── Resume: load already-completed jobs from plan_rankings.csv only ────────
        # A job is considered complete only if every plan file on disk for that job
        # already has a matching row in the rankings CSV.
        completed_jobs = set()
        all_runtime_list = []
        all_ranking_list = []

        if os.path.exists(ranking_df_path):
            existing_ranking_df = pd.read_csv(ranking_df_path, header=0)
            all_ranking_list = existing_ranking_df.to_dict('records')
            print(f'[RESUME] Found existing rankings CSV: {ranking_df_path}')

            # Build a dict of {job -> set of plan filenames} already recorded
            ranked_plans_per_job = (
                existing_ranking_df.groupby('Job')['Plan']
                .apply(set)
                .to_dict()
            )

            # Verify completeness: all plan files on disk must be in the rankings
            for candidate_job, ranked_plans in ranked_plans_per_job.items():
                if targeted_enum_method == "DB":
                    candidate_plan_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/enumerated_plans_{targeted_enum_method}/{query_folder_name}/iter{iter}/{candidate_job}/"
                else:
                    plan_folder_postfix = "_".join([targeted_enum_method, targeted_model_name])
                    candidate_plan_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/enumerated_plans_{plan_folder_postfix}/{query_folder_name}/iter{iter}/{candidate_job}/"

                if not os.path.exists(candidate_plan_path):
                    print(f'[RESUME] Plan folder missing for {candidate_job} — will re-run.')
                    continue

                plans_on_disk = set(
                    f for f in os.listdir(candidate_plan_path) if f.endswith(".txt")
                )
                if plans_on_disk and plans_on_disk == ranked_plans:
                    completed_jobs.add(candidate_job)
                else:
                    missing = plans_on_disk - ranked_plans
                    extra = ranked_plans - plans_on_disk
                    print(f'[RESUME] {candidate_job} incomplete in rankings '
                          f'(missing={missing}, unexpected={extra}) — will re-run.')

            print(f'[RESUME] {len(completed_jobs)} fully-completed jobs: {sorted(completed_jobs)}')
        else:
            print('[RESUME] No existing rankings CSV found — starting fresh.')

        # Pre-populate all_runtime_list from the runtime CSV if it exists,
        # but only for jobs confirmed complete via the rankings check above.
        if os.path.exists(all_df_path):
            existing_runtime_df = pd.read_csv(all_df_path, header=0, index_col=0)
            for completed_job in completed_jobs:
                if completed_job in existing_runtime_df['Job'].values:
                    job_slice = existing_runtime_df[existing_runtime_df['Job'] == completed_job].copy()
                    job_slice.index = [targeted_enum_method] * len(job_slice)
                    all_runtime_list.append(job_slice)
            print(f'[RESUME] Loaded runtime rows for {len(all_runtime_list)} completed jobs from: {all_df_path}')
        else:
            print('[RESUME] No existing runtime CSV found — will be created fresh.')

        # ── Main runtime loop ────────────────────────────────────────────────────
        for job_index, job in enumerate(testquery):
            if job in ["Job1461v3"]:
                continue
            if targeted_test_query == "tpch-d" and job in ["Job17v3", "Job1131v3", "Job128v3"]:
                continue
            if targeted_test_query == "tpch-l" and job not in ['Job49v1', 'Job44v3', 'Job25v1', 'Job253v4', 'Job253v0',
                                                               'Job185v3', 'Job185v2', 'Job16v0', 'Job161v0', 'Job1325v3',
                                                               'Job1300v2', 'Job128v4', 'Job128v3', 'Job128v0', 'Job1252v1',
                                                               'Job1103v3', 'Job1034v4', 'Job1034v0', 'Job1033v4', 'Job1028v2',
                                                               'Job1021v1', 'Job1017v3', 'Job1017v1', 'Job1011v2']:
                continue

            # ── RESUME CHECK: skip jobs whose results are already saved ──────────
            if job in completed_jobs:
                print(f'[RESUME] Skipping already-completed job ({job_index}): {job}')
                continue

            print('run job: ', job_index, job)
            print('Job Start time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            if targeted_enum_method == "DB":
                plan_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/enumerated_plans_{targeted_enum_method}/{query_folder_name}/iter{iter}/{job}/"
            else:
                plan_folder_postfix = "_".join([targeted_enum_method, targeted_model_name])
                plan_path = f"/Users/grant/Documents/ITU/Spring26/CodeBase/LTR_cross_hardwares/results/enumerated_plans_{plan_folder_postfix}/{query_folder_name}/iter{iter}/{job}/"

            plan_files = sorted(
                [f for f in os.listdir(plan_path) if f.endswith(".txt")],
                key=lambda x: int(x.split(".")[0])
            )
            print(f"Found {len(plan_files)} plans for job {job}: {plan_files}")

            def _execute_plan(plan_sql):
                cursor.execute("SET STATISTICS TIME ON")
                cursor.execute(plan_sql)
                cpu, time_val, cpu_unit, time_unit = None, None, 'ms', 'ms'
                while cursor.nextset():
                    mess = cursor.messages
                    parts = mess[0][1].split(",")
                    if len(parts) == 3:
                        cpu = int(parts[1].split("=")[1][:-3])
                        cpu_unit = parts[1].split("=")[1][-3:]
                        time_val = int(parts[2].split("=")[1][:-3])
                        time_unit = parts[2].split("=")[1][-3:]
                    elif len(parts) == 2:
                        cpu = int(parts[0].split("=")[1][:-3])
                        cpu_unit = parts[0].split("=")[1][-3:]
                        time_val = int(parts[1].split("=")[1][:-3])
                        time_unit = parts[1].split("=")[1][-3:]
                    else:
                        print('mess', mess)
                return cpu, time_val, cpu_unit, time_unit

            methods_runtime_list = []
            plan_time_records = []

            for plan_file in plan_files:
                with open(plan_path + plan_file, "r") as f:
                    plan_sql = f.read()
                try:
                    cpu, time_val, cpu_unit, time_unit = _execute_plan(plan_sql)
                    if cpu is None:
                        print(f"  Could not parse timing for {plan_file}, skipping.")
                        continue
                    result = [job, plan_file, cpu, time_val, cpu + time_val, cpu_unit, time_unit]
                    plan_time_records.append((plan_file, cpu, time_val, cpu + time_val, cpu_unit, time_unit))
                except OperationalError as err:
                    print(f"  Timeout on {plan_file}")
                    print(job, targeted_enum_method, targeted_model_name, err)
                    result = [job, plan_file, -1, -1, -1, 'ms', 'ms']
                    plan_time_records.append((plan_file, -1, -1, -1, 'ms', 'ms'))
                    print("----")
                except ProgrammingError as err:
                    print(f"  ProgrammingError on {plan_file}")
                    print(job, targeted_enum_method, targeted_model_name, err)
                    if targeted_enum_method != "DB":
                        result = [job, plan_file, -2, -2, -2, 'ms', 'ms']
                        plan_time_records.append((plan_file, -2, -2, -2, 'ms', 'ms'))
                        print("----")
                    else:
                        pure_sql = plan_sql.split("OPTION")[0] + " OPTION (MAXDOP 1) "
                        cpu, time_val, cpu_unit, time_unit = _execute_plan(pure_sql)
                        result = [job, plan_file, cpu, time_val, cpu + time_val, cpu_unit, time_unit]
                        plan_time_records.append((plan_file, cpu, time_val, cpu + time_val, cpu_unit, time_unit))
                except Exception as exce:
                    print(f"  Exception on {plan_file}:", exce)
                    continue
                print(f"  statistic [{plan_file}]:", result)
                methods_runtime_list.append(result)

            print('Job End time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            if methods_runtime_list:
                job_df = pd.DataFrame(methods_runtime_list)
                job_df.columns = "Job,Plan,CPU time,Time,Sum,CPU unit,Time unit".split(',')
                job_df.index = [targeted_enum_method] * len(job_df)
                all_runtime_list.append(job_df)

            if plan_time_records:
                def _sort_key(record):
                    t = record[2]
                    return t if t >= 0 else float('inf')
                sorted_plans = sorted(plan_time_records, key=_sort_key)
                for rank, (plan_file, cpu, time_val, sum_val, cpu_unit, time_unit) in enumerate(sorted_plans, start=1):
                    all_ranking_list.append({
                        'Job': job,
                        'Rank': rank,
                        'Plan': plan_file,
                        'CPU time': cpu,
                        'Time': time_val,
                        'Sum': sum_val,
                        'CPU unit': cpu_unit,
                        'Time unit': time_unit,
                    })
                print(f"  Rankings for {job}: {[(r, p) for r, (p, *_) in enumerate(sorted_plans, 1)]}")

            # ── Checkpoint: write both CSVs after each query ─────────────────────
            if all_runtime_list:
                all_runtime_df = pd.concat(all_runtime_list, axis=0)
                all_runtime_df.to_csv(all_df_path, index=True, header=True)
            if all_ranking_list:
                ranking_df = pd.DataFrame(all_ranking_list)
                ranking_df.to_csv(ranking_df_path, index=False, header=True)

        ### describe the runtime distributions of test queries
        if all_runtime_list:
            all_runtime_df = pd.concat(all_runtime_list, axis=0)
            runtime_desc_list = []
            for method in [targeted_enum_method]:
                method_runtime_df = all_runtime_df[all_runtime_df.index == method]
                method_runtime_desc_df = method_runtime_df['Time'].describe()
                runtime_desc_list.append(method_runtime_desc_df)
            runtime_desc_df = pd.concat(runtime_desc_list, axis=1)
            runtime_desc_df.columns = [targeted_enum_method]
            runtime_desc_df = runtime_desc_df.round(1)
            runtime_desc_df.to_csv(runtime_desc_file_path, index=True, header=True)

    print('end time: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    End_time = datetime.now()
    Elapsed_time = round((End_time - Start_time).total_seconds() / 60, 2)
    print('Total Elapsed Time: ', Elapsed_time)