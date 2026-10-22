import select_runs as sr

df = sr.load_parsing_overview("BLU")            # parsing overview, version v0
ped = sr.load_pedestal_outliers("BLU")          # pedestal summary, version v1, or None
mon = sr.load_monitoring_scan("BLU")            # monitoring rates, or None

good_runs, details, cutflow, trigger_info = sr.select_runs(
    df, ped, mon, run_type="ADC", trigger_rate_n_sigma=3.0, label="[BLU]")

print(good_runs)