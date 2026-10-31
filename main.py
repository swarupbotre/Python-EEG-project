
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_FILE = "expanded_sleep_data_with_bio.csv"
OUTPUT_DIR = Path("project_outputs")

EEG_COLUMNS = ["EEG_Alpha", "EEG_Beta", "EEG_Theta"]

SLEEP_COLUMNS = [
    "Sleep Duration",
    "Quality of Sleep",
    "Stress Level",
    "Physical Activity Level",
    "Daily Steps",
]

NUMERIC_ANALYSIS_COLUMNS = EEG_COLUMNS + SLEEP_COLUMNS + [
    "Age",
    "Heart Rate",
    "ECG_HeartRate",
    "ECG_QRS_Duration",
    "ECF_Na",
    "ECF_K",
    "EMG_Mean",
    "EMG_Max",
]


def load_data(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find '{file_path}'. Put the CSV in the same folder as main.py."
        )

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].astype("string").str.strip()

    return data


def explore_data(data):
    print("\n" + "=" * 65)
    print("1. DATASET OVERVIEW")
    print("=" * 65)

    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values in each column:")
    print(data.isnull().sum())

    print(f"\nDuplicate rows: {data.duplicated().sum()}")

    print("\nNumerical summary:")
    print(data.describe().round(2).to_string())

    print("\nSleep disorder counts:")

    if "Sleep Disorder" in data.columns:
        print(data["Sleep Disorder"].value_counts(dropna=False))


def clean_data(data):
    cleaned = data.copy()

    for column in NUMERIC_ANALYSIS_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column], errors="coerce"
            )

    if "BMI Category" in cleaned.columns:
        cleaned["BMI Category"] = cleaned["BMI Category"].replace(
            {"Normal Weight": "Normal"}
        )

    if "Blood Pressure" in cleaned.columns:
        bp_parts = cleaned["Blood Pressure"].astype("string").str.split(
            "/", n=1, expand=True
        )

        if bp_parts.shape[1] == 2:
            cleaned["Systolic BP"] = pd.to_numeric(
                bp_parts[0], errors="coerce"
            )
            cleaned["Diastolic BP"] = pd.to_numeric(
                bp_parts[1], errors="coerce"
            )

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    removed = before - len(cleaned)

    print("\n" + "=" * 65)
    print("2. DATA CLEANING")
    print("=" * 65)

    print(f"Duplicate rows removed: {removed}")
    print("\nMissing values after conversion/cleaning:")
    print(cleaned.isnull().sum())

    return cleaned


def descriptive_analysis(data):
    print("\n" + "=" * 65)
    print("3. DESCRIPTIVE STATISTICS")
    print("=" * 65)

    available = [
        column
        for column in EEG_COLUMNS + SLEEP_COLUMNS
        if column in data.columns
    ]

    summary = data[available].agg(
        ["mean", "std", "min", "max"]
    ).T.round(3)

    print("\nMean, standard deviation, minimum and maximum:")
    print(summary.to_string())

    if "Sleep Disorder" in data.columns:
        print("\nMean EEG measurements by sleep-disorder category:")

        grouped = data.groupby("Sleep Disorder")[EEG_COLUMNS].mean().round(2)
        print(grouped.to_string())

        print("\nMean sleep measures by sleep-disorder category:")

        sleep_grouped = data.groupby("Sleep Disorder")[
            ["Sleep Duration", "Quality of Sleep", "Stress Level"]
        ].mean().round(2)

        print(sleep_grouped.to_string())

    return summary


def correlation_analysis(data):
    print("\n" + "=" * 65)
    print("4. CORRELATION ANALYSIS (SPEARMAN)")
    print("=" * 65)

    columns = [
        column
        for column in EEG_COLUMNS + SLEEP_COLUMNS
        if column in data.columns
    ]

    correlation = data[columns].corr(method="spearman")

    print("\nSpearman correlation matrix:")
    print(correlation.round(3).to_string())

    print("\nSelected EEG-to-sleep correlations:")

    for eeg_column in EEG_COLUMNS:
        for other_column in [
            "Sleep Duration",
            "Quality of Sleep",
            "Stress Level",
        ]:
            if (
                eeg_column in data.columns
                and other_column in data.columns
            ):
                pair = data[[eeg_column, other_column]].dropna()

                if len(pair) >= 3:
                    rho, p_value = stats.spearmanr(
                        pair[eeg_column], pair[other_column]
                    )

                    print(
                        f"{eeg_column} vs {other_column}: "
                        f"rho={rho:.3f}, p-value={p_value:.4g}, "
                        f"n={len(pair)}"
                    )

    return correlation


def kruskal_group_tests(data):
    print("\n" + "=" * 65)
    print("5. GROUP COMPARISON (KRUSKAL-WALLIS TEST)")
    print("=" * 65)

    if "Sleep Disorder" not in data.columns:
        print("Sleep Disorder column not found; tests skipped.")
        return []

    results = []

    for column in EEG_COLUMNS + [
        "Quality of Sleep",
        "Sleep Duration",
    ]:
        if column not in data.columns:
            continue

        groups = []
        group_names = []

        for name, group in data.groupby("Sleep Disorder"):
            values = pd.to_numeric(
                group[column], errors="coerce"
            ).dropna()

            if len(values) > 0:
                groups.append(values.to_numpy())
                group_names.append(name)

        if len(groups) >= 2:
            statistic, p_value = stats.kruskal(*groups)

            results.append(
                (column, statistic, p_value, group_names)
            )

            conclusion = (
                "statistically significant difference detected"
                if p_value < 0.05
                else "no statistically significant difference detected"
            )

            print(
                f"{column}: H={statistic:.3f}, "
                f"p-value={p_value:.4g} -> {conclusion} "
                "at alpha=0.05"
            )

    print(
        "\nKruskal-Wallis tests whether at least one group differs. "
        "It does not identify which pairs differ."
    )

    return results


def create_visualizations(data, correlation):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme()

    print("\n" + "=" * 65)
    print("6. CREATING VISUALIZATIONS")
    print("=" * 65)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, column in zip(axes, EEG_COLUMNS):
        sns.histplot(data=data, x=column, kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_eeg_distributions.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )

    ax.set_title("Spearman Correlation: EEG and Sleep Variables")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.show()
    plt.close(fig)

    melted = data.melt(
        id_vars=["Sleep Disorder"],
        value_vars=EEG_COLUMNS,
        var_name="EEG Feature",
        value_name="Measurement",
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(
        data=melted,
        x="Sleep Disorder",
        y="Measurement",
        hue="EEG Feature",
        ax=ax,
    )

    ax.set_title("EEG Measurements Across Sleep-Disorder Categories")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_eeg_by_sleep_disorder.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, column in zip(axes, EEG_COLUMNS):
        sns.regplot(
            data=data,
            x="Quality of Sleep",
            y=column,
            scatter_kws={"alpha": 0.25, "s": 15},
            line_kws={"color": "red"},
            ax=ax,
        )

        ax.set_title(f"{column} vs Quality of Sleep")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_eeg_vs_sleep_quality.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, column in zip(axes, EEG_COLUMNS):
        sns.boxplot(
            data=data,
            x="Stress Level",
            y=column,
            ax=ax,
        )

        ax.set_title(f"{column} by Stress Level")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_eeg_by_stress.png", dpi=150)
    plt.show()
    plt.close(fig)

    means = data.groupby("Sleep Disorder")[EEG_COLUMNS].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    means.plot(kind="bar", ax=ax)

    ax.set_title("Average EEG Measurements by Sleep-Disorder Category")
    ax.set_ylabel("Mean measurement")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_average_eeg_by_disorder.png", dpi=150)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(
        data=data,
        x="Sleep Duration",
        y="EEG_Alpha",
        hue="Sleep Disorder",
        alpha=0.7,
        ax=ax,
    )

    ax.set_title("EEG Alpha vs Sleep Duration")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_alpha_vs_sleep_duration.png", dpi=150)
    plt.show()
    plt.close(fig)

    print(f"\nSaved graphs in: {OUTPUT_DIR.resolve()}")


def save_results(data, summary, correlation, test_results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        OUTPUT_DIR / "cleaned_sleep_data.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv"
    )

    correlation.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv"
    )

    if test_results:
        test_table = pd.DataFrame(
            [
                {
                    "Variable": variable,
                    "Kruskal_Wallis_H": statistic,
                    "p_value": p_value,
                    "Groups": ", ".join(groups),
                }
                for variable, statistic, p_value, groups in test_results
            ]
        )

        test_table.to_csv(
            OUTPUT_DIR / "kruskal_wallis_results.csv",
            index=False,
        )

    print("\nSaved cleaned data and result tables in the output folder.")


def main():
    print("=" * 65)
    print("EEG AND SLEEP HEALTH ANALYSIS")
    print("=" * 65)

    try:
        data = load_data(DATA_FILE)
    except FileNotFoundError as error:
        print(error)
        print(
            "Move the CSV file into the same folder as main.py "
            "and run again."
        )
        return

    explore_data(data)
    cleaned = clean_data(data)
    summary = descriptive_analysis(cleaned)
    correlation = correlation_analysis(cleaned)
    test_results = kruskal_group_tests(cleaned)

    create_visualizations(cleaned, correlation)

    save_results(
        cleaned,
        summary,
        correlation,
        test_results,
    )

    print("\n" + "=" * 65)
    print("PROJECT RUN COMPLETED")
    print("=" * 65)
    print(
        "Check the 'project_outputs' folder for graphs, "
        "cleaned CSV, and tables."
    )
    print("Association does not prove causation.")


if __name__ == "__main__":
    main()
