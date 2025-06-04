import pandas as pd
from scipy.stats import ttest_rel
from pathlib import Path

def analyze_model_performance(csv_path: str, dataset_name: str, arch: str):
    df = pd.read_csv(csv_path)

    df_shield = df[df.model == "ShieldedMLP"].sort_values("trial")
    df_kkt = df[df.model == "ShieldedMLPWithKKTSTE"].sort_values("trial")

    if len(df_shield) != len(df_kkt):
        raise ValueError("Mismatched number of trials between Shielded and KKTSTE")

    shield_rmse = df_shield["test_rmse"].values
    kkt_rmse = df_kkt["test_rmse"].values

    delta_rmse = kkt_rmse - shield_rmse
    t_stat, p_val = ttest_rel(kkt_rmse, shield_rmse)
    cohens = delta_rmse.mean() / delta_rmse.std(ddof=1)

    result = {
        "Dataset": dataset_name,
        "Architecture": arch,
        "Shielded Mean": round(shield_rmse.mean(), 6),
        "KKTSTE Mean": round(kkt_rmse.mean(), 6),
        "ΔRMSE (KKT - Shield)": round(delta_rmse.mean(), 6),
        "t-statistic": round(t_stat, 6),
        "p-value": round(p_val, 6),
        "Significant @ 0.05": p_val < 0.05,
        "Cohen's d": round(cohens, 6)
    }

    return result


def main():
    datasets = {
        "URL": "out/url",
        "Faulty Steel Plates": "out/faulty-steel-plates",
        "LCLD": "out/lcld",
        "News": "out/news"
    }

    architectures = {
        "Shallow": "final_rmses_shallow.csv",
        "Deep": "final_rmses_deep.csv"
    }

    results = []
    for dataset_name, base_path in datasets.items():
        for arch_name, file_name in architectures.items():
            try:
                csv_path = Path(base_path) / file_name
                results.append(analyze_model_performance(csv_path, dataset_name, arch_name))
            except Exception as e:
                print(f"[ERROR] {dataset_name} ({arch_name}): {e}")

    df_out = pd.DataFrame(results)
    output_path = Path("out/kkt_vs_shield_summary.csv")
    df_out.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.resolve()}")
    print("\n=== Summary ===")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()
