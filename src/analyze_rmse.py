import pandas as pd
from scipy.stats import ttest_rel
from pathlib import Path

def analyze_model_performance(csv_path: str, dataset_name: str, arch: str, method: str):
    df = pd.read_csv(csv_path)

    df_shield = df[df.model == "ShieldedMLP"].sort_values("trial")
    
    if method == "Projected":
        df_other = df[df.model == "ShieldWithProjGrad"].sort_values("trial")
    elif method == "Masked":
        df_other = df[df.model == "ShieldedMLPMasked"].sort_values("trial")
    else:
        raise ValueError(f"Unknown method type: {method}")

    if len(df_shield) != len(df_other):
        raise ValueError(f"Mismatched number of trials between Shielded and {method} for {dataset_name} - {arch}")

    shield_rmse = df_shield["test_rmse"].values
    other_rmse = df_other["test_rmse"].values

    shield_avg_time = df_shield["total_time"].mean()
    other_avg_time = df_other["total_time"].mean()

    improvement = other_rmse - shield_rmse
    percent_change = (improvement.mean() / shield_rmse.mean()) * 100

    t_stat, p_val = ttest_rel(other_rmse, shield_rmse)
    cohens = improvement.mean() / improvement.std(ddof=1)

    result = {
        "Dataset": dataset_name,
        "Architecture": arch,
        "Method": method,
        "Shielded Mean RMSE": round(shield_rmse.mean(), 4),
        "Method Mean RMSE": round(other_rmse.mean(), 4),
        "Percent Change (Method - Shield%)": round(percent_change, 2),
        "t-statistic": round(t_stat, 3),
        "p-value": round(p_val, 4),
        "Significant @ 0.05": p_val < 0.05,
        "Cohen's d": round(cohens, 3),
        "Shielded Avg Time (s)": round(shield_avg_time, 2),
        "Method Avg Time (s)": round(other_avg_time, 2),
    }

    return result


def main():
    datasets = {
        "URL": "out/url",
        "Faulty Steel Plates": "out/faulty-steel-plates",
        "LCLD": "out/lcld",
        "News": "out/news",
    }

    architectures = {
        "Shallow": ("final_rmses_shallow.csv", "final_rmses_masked_shallow.csv"),
        "Deep": ("final_rmses_deep.csv", "final_rmses_masked_deep.csv"),
    }

    results = []

    for dataset_name, base_path in datasets.items():
        for arch_name, (unmasked_file, _) in architectures.items():
            csv_path = Path(base_path) / unmasked_file
            try:
                results.append(analyze_model_performance(csv_path, dataset_name, arch_name, method="Projected"))
            except Exception as e:
                print(f"[ERROR] {dataset_name} ({arch_name} Projected): {e}")

    for dataset_name, base_path in datasets.items():
        if dataset_name == "LCLD":
            continue
        for arch_name, (_, masked_file) in architectures.items():
            csv_path = Path(base_path) / masked_file
            try:
                results.append(analyze_model_performance(csv_path, dataset_name, arch_name, method="Masked"))
            except Exception as e:
                print(f"[ERROR] {dataset_name} ({arch_name} Masked): {e}")

    df_out = pd.DataFrame(results)
    output_path = Path("out/projected_vs_masked_summary.csv")
    df_out.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.resolve()}")
    print("\n=== Summary ===")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
