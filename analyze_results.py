import pandas as pd
from scipy.stats import ttest_rel
from pathlib import Path

def analyze_model_performance(csv_path: str, dataset_name: str):
    df = pd.read_csv(csv_path)

    df_shield = df[df.model == "ShieldedMLP"].sort_values("trial")
    df_kkt = df[df.model == "ShieldedMLPWithKKTSTE"].sort_values("trial")

    if len(df_shield) != len(df_kkt):
        raise ValueError("Mismatched number of trials between Shielded and KKTSTE")

    shield_rmse = df_shield["test_rmse"].values
    kkt_rmse = df_kkt["test_rmse"].values

    delta_rmse = kkt_rmse - shield_rmse
    t_stat, p_val = ttest_rel(kkt_rmse, shield_rmse)

    result = {
        "Dataset": dataset_name,
        "Shielded Mean": round(shield_rmse.mean(), 4),
        "KKTSTE Mean": round(kkt_rmse.mean(), 4),
        "ΔRMSE (KKT - Shield)": round(delta_rmse.mean(), 4),
        "t-statistic": round(t_stat, 4),
        "p-value": round(p_val, 4),
        "Significant @ 0.05": p_val < 0.05
    }

    return result

def main():
    datasets = {
        "URL": "out/url/final_rmses.csv",
        "Faulty Steel Plates": "out/faulty-steel-plates/final_rmses.csv",
        "LCLD": "out/lcld/final_rmses.csv",
        "News": "out/news/final_rmses.csv"
    }

    results = []
    for name, path in datasets.items():
        try:
            results.append(analyze_model_performance(path, name))
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    df_out = pd.DataFrame(results)
    output_path = Path("out/kkt_vs_shield_summary.csv")
    df_out.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.resolve()}")
    print("\n=== Summary ===")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()
