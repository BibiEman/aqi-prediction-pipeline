from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_training_dataset.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "eda"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_data():
    print("=" * 70)
    print("AQI EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(f"\nLoading dataset:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print("\nDataset loaded successfully.")

    print(
        f"\nShape: {df.shape}"
    )

    print(
        f"Date range: "
        f"{df['timestamp'].min()} "
        f"to "
        f"{df['timestamp'].max()}"
    )

    print(
        f"Cities: "
        f"{df['city'].nunique()}"
    )

    print("\nColumns:")

    for column in df.columns:
        print(f"  - {column}")

    return df

def analyze_data_quality(df):
    
    """
    Analyze missing values, duplicates, data types,
    AQI distribution, and descriptive statistics.
    """

    print("\n" + "=" * 70)
    print("DATA QUALITY AND DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(
        f"\nDuplicate rows: {duplicate_count:,}"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    print("\nMissing values:")

    if missing.empty:
        print("No missing values detected.")
    else:
        print(
            missing.to_string()
        )

    # --------------------------------------------------------
    # Dataset data types
    # --------------------------------------------------------

    print("\nData types:")
    print(
        df.dtypes.to_string()
    )

    # --------------------------------------------------------
    # Target AQI statistics
    # --------------------------------------------------------

    print("\nTarget AQI statistics:")

    print(
        df["target_aqi"]
        .describe()
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # Current AQI statistics
    # --------------------------------------------------------

    print("\nCurrent AQI statistics:")

    print(
        df["us_aqi"]
        .describe()
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # AQI statistics by city
    # --------------------------------------------------------

    city_stats = (
        df.groupby("city")["us_aqi"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
                "std",
            ]
        )
        .round(2)
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    print("\nAQI statistics by city:")
    print(
        city_stats.to_string()
    )

    # --------------------------------------------------------
    # Save city statistics
    # --------------------------------------------------------

    city_stats.to_csv(
        RESULTS_DIR
        / "city_aqi_statistics.csv"
    )

    print(
        "\nCity AQI statistics saved to:"
    )

    print(
        RESULTS_DIR
        / "city_aqi_statistics.csv"
    )

    return city_stats

def plot_aqi_distribution(df):
    """
    Visualize the overall distribution of current AQI values.
    """

    print("\n" + "=" * 70)
    print("AQI DISTRIBUTION ANALYSIS")
    print("=" * 70)

    output_path = (
        RESULTS_DIR
        / "aqi_distribution.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        df["us_aqi"],
        bins=40,
        edgecolor="black",
        alpha=0.8,
    )

    mean_aqi = df["us_aqi"].mean()
    median_aqi = df["us_aqi"].median()

    plt.axvline(
        mean_aqi,
        linestyle="--",
        linewidth=2,
        label=f"Mean AQI = {mean_aqi:.1f}",
    )

    plt.axvline(
        median_aqi,
        linestyle=":",
        linewidth=2,
        label=f"Median AQI = {median_aqi:.1f}",
    )

    plt.title(
        "Distribution of AQI Across Pakistan Cities",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("US AQI")
    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nMean AQI   : {mean_aqi:.2f}"
    )

    print(
        f"Median AQI : {median_aqi:.2f}"
    )

    print(
        f"Minimum AQI: {df['us_aqi'].min():.2f}"
    )

    print(
        f"Maximum AQI: {df['us_aqi'].max():.2f}"
    )

    print(
        f"\nAQI distribution plot saved to:\n"
        f"{output_path}"
    )

def plot_city_aqi_comparison(df):
    """
    Compare average AQI levels across all monitored cities.
    """

    print("\n" + "=" * 70)
    print("CITY-WISE AQI COMPARISON")
    print("=" * 70)

    output_path = (
        RESULTS_DIR
        / "city_average_aqi.png"
    )

    city_mean = (
        df.groupby("city")["us_aqi"]
        .mean()
        .sort_values(
            ascending=True
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.barh(
        city_mean.index,
        city_mean.values,
    )

    plt.title(
        "Average AQI by City",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("Average US AQI")
    plt.ylabel("City")

    for bar in bars:
        width = bar.get_width()

        plt.text(
            width + 1,
            bar.get_y()
            + bar.get_height() / 2,
            f"{width:.1f}",
            va="center",
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nAverage AQI by city:")

    print(
        city_mean
        .sort_values(
            ascending=False
        )
        .round(2)
        .to_string()
    )

    print(
        f"\nCity comparison plot saved to:\n"
        f"{output_path}"
    )


def plot_temporal_aqi_trends(df):
    """
    Analyze daily and monthly AQI trends by city.
    """

    print("\n" + "=" * 70)
    print("TEMPORAL AQI TREND ANALYSIS")
    print("=" * 70)

    output_path = (
        RESULTS_DIR
        / "daily_aqi_trends_by_city.png"
    )

    # --------------------------------------------------------
    # Calculate daily mean AQI
    # --------------------------------------------------------

    temporal_df = df.copy()

    temporal_df["date"] = (
        temporal_df["timestamp"]
        .dt.date
    )

    daily_aqi = (
        temporal_df
        .groupby(
            ["date", "city"],
            as_index=False,
        )["us_aqi"]
        .mean()
    )

    daily_aqi["date"] = pd.to_datetime(
        daily_aqi["date"]
    )

    # --------------------------------------------------------
    # Plot daily trends
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 8)
    )

    for city in sorted(
        daily_aqi["city"].unique()
    ):
        city_data = daily_aqi[
            daily_aqi["city"] == city
        ]

        plt.plot(
            city_data["date"],
            city_data["us_aqi"],
            linewidth=1.5,
            label=city,
        )

    plt.title(
        "Daily Average AQI Trends by City",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("Date")
    plt.ylabel("Daily Average US AQI")

    plt.legend(
        title="City",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Monthly statistics
    # --------------------------------------------------------

    monthly_aqi = (
        df.groupby(
            ["month", "city"],
            as_index=False,
        )["us_aqi"]
        .mean()
    )

    monthly_table = (
        monthly_aqi
        .pivot(
            index="month",
            columns="city",
            values="us_aqi",
        )
        .round(2)
    )

    monthly_output_path = (
        RESULTS_DIR
        / "monthly_aqi_by_city.csv"
    )

    monthly_table.to_csv(
        monthly_output_path
    )

    print("\nMonthly average AQI by city:")

    print(
        monthly_table.to_string()
    )

    print(
        "\nDaily temporal trend plot saved to:"
    )

    print(
        output_path
    )

    print(
        "\nMonthly AQI statistics saved to:"
    )

    print(
        monthly_output_path
    )

def check_city_series_similarity(df):
    """
    Check whether Islamabad and Rawalpindi contain identical
    AQI observations across the historical dataset.
    """

    print("\n" + "=" * 70)
    print("CITY SERIES SIMILARITY CHECK")
    print("=" * 70)

    comparison = (
        df[df["city"].isin(["Islamabad", "Rawalpindi"])]
        .pivot(
            index="timestamp",
            columns="city",
            values="us_aqi",
        )
        .dropna()
    )

    identical_count = (
        comparison["Islamabad"]
        == comparison["Rawalpindi"]
    ).sum()

    total_count = len(comparison)

    identical_percentage = (
        identical_count
        / total_count
        * 100
    )

    max_difference = (
        comparison["Islamabad"]
        - comparison["Rawalpindi"]
    ).abs().max()

    print(
        f"\nComparable timestamps : {total_count:,}"
    )

    print(
        f"Identical AQI values  : {identical_count:,}"
    )

    print(
        f"Identical percentage  : "
        f"{identical_percentage:.2f}%"
    )

    print(
        f"Maximum difference    : "
        f"{max_difference:.2f}"
    )

def analyze_feature_correlations(df):
    """
    Analyze correlations between AQI and major pollutant,
    weather, and engineered numerical features.
    """

    print("\n" + "=" * 70)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 70)

    selected_features = [
        "us_aqi",
        "target_aqi",
        "pm2_5",
        "pm10",
        "ozone",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "carbon_monoxide",
        "temperature",
        "humidity",
        "pressure",
        "cloud_cover",
        "wind_speed",
        "precipitation",
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_24",
        "aqi_roll_3",
        "aqi_roll_6",
        "aqi_roll_24",
    ]

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    correlation_matrix = (
        df[selected_features]
        .corr()
    )

    correlation_output = (
        RESULTS_DIR
        / "feature_correlation_matrix.csv"
    )

    correlation_matrix.to_csv(
        correlation_output
    )

    # --------------------------------------------------------
    # Correlation with target AQI
    # --------------------------------------------------------

    target_correlations = (
        correlation_matrix["target_aqi"]
        .drop("target_aqi")
        .sort_values(
            key=lambda values: values.abs(),
            ascending=False,
        )
    )

    print(
        "\nFeature correlations with target AQI:"
    )

    print(
        target_correlations
        .round(4)
        .to_string()
    )

    target_correlations.to_csv(
        RESULTS_DIR
        / "target_aqi_correlations.csv",
        header=[
            "Correlation_With_Target_AQI"
        ],
    )

    # --------------------------------------------------------
    # Correlation heatmap
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 11)
    )

    image = plt.imshow(
        correlation_matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Pearson Correlation",
    )

    plt.xticks(
        range(len(selected_features)),
        selected_features,
        rotation=90,
        fontsize=8,
    )

    plt.yticks(
        range(len(selected_features)),
        selected_features,
        fontsize=8,
    )

    plt.title(
        "Correlation Matrix of AQI, Pollutants, Weather and Lag Features",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    heatmap_path = (
        RESULTS_DIR
        / "feature_correlation_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nCorrelation matrix saved to:"
    )

    print(
        correlation_output
    )

    print(
        "\nCorrelation heatmap saved to:"
    )

    print(
        heatmap_path
    )

def analyze_hourly_aqi_pattern(df):
    """
    Analyze how average AQI changes throughout the day.
    """

    print("\n" + "=" * 70)
    print("HOURLY AQI PATTERN ANALYSIS")
    print("=" * 70)

    hourly_aqi = (
        df.groupby("hour")["us_aqi"]
        .agg(
            [
                "mean",
                "median",
                "std",
                "count",
            ]
        )
        .round(2)
    )

    print("\nAQI statistics by hour:")
    print(
        hourly_aqi.to_string()
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "hourly_aqi_statistics.csv"
    )

    hourly_aqi.to_csv(
        csv_path
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        hourly_aqi.index,
        hourly_aqi["mean"],
        marker="o",
        linewidth=2,
        label="Mean AQI",
    )

    plt.plot(
        hourly_aqi.index,
        hourly_aqi["median"],
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label="Median AQI",
    )

    plt.title(
        "Average AQI by Hour of Day",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("US AQI")

    plt.xticks(
        range(0, 24, 2)
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        RESULTS_DIR
        / "hourly_aqi_pattern.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Identify highest and lowest AQI hours
    # --------------------------------------------------------

    highest_hour = (
        hourly_aqi["mean"]
        .idxmax()
    )

    lowest_hour = (
        hourly_aqi["mean"]
        .idxmin()
    )

    print(
        f"\nHighest average AQI hour: "
        f"{highest_hour:02d}:00 "
        f"({hourly_aqi.loc[highest_hour, 'mean']:.2f})"
    )

    print(
        f"Lowest average AQI hour : "
        f"{lowest_hour:02d}:00 "
        f"({hourly_aqi.loc[lowest_hour, 'mean']:.2f})"
    )

    print(
        "\nHourly AQI statistics saved to:"
    )

    print(
        csv_path
    )

    print(
        "\nHourly AQI plot saved to:"
    )

    print(
        plot_path
    )

def analyze_aqi_categories(df):
    """
    Analyze the frequency of historical AQI categories.
    """

    print("\n" + "=" * 70)
    print("AQI CATEGORY ANALYSIS")
    print("=" * 70)

    analysis_df = df.copy()

    bins = [
        -np.inf,
        50,
        100,
        150,
        200,
        300,
        np.inf,
    ]

    labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]

    analysis_df["aqi_category"] = pd.cut(
        analysis_df["us_aqi"],
        bins=bins,
        labels=labels,
        right=True,
    )

    # --------------------------------------------------------
    # Overall category counts
    # --------------------------------------------------------

    category_counts = (
        analysis_df["aqi_category"]
        .value_counts(sort=False)
    )

    category_percentage = (
        category_counts
        / len(analysis_df)
        * 100
    ).round(2)

    category_summary = pd.DataFrame(
        {
            "Count": category_counts,
            "Percentage": category_percentage,
        }
    )

    print("\nOverall AQI category distribution:")

    print(
        category_summary.to_string()
    )

    # --------------------------------------------------------
    # Save category statistics
    # --------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "aqi_category_distribution.csv"
    )

    category_summary.to_csv(
        csv_path
    )

    # --------------------------------------------------------
    # City-wise hazardous observations
    # --------------------------------------------------------

    hazardous_by_city = (
        analysis_df[
            analysis_df["aqi_category"]
            == "Hazardous"
        ]
        .groupby("city")
        .size()
        .reindex(
            sorted(
                analysis_df["city"].unique()
            ),
            fill_value=0,
        )
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nHazardous observations by city (AQI > 300):"
    )

    print(
        hazardous_by_city.to_string()
    )

    hazardous_by_city.to_csv(
        RESULTS_DIR
        / "hazardous_observations_by_city.csv",
        header=["Hazardous_Count"],
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    bars = plt.bar(
        category_summary.index.astype(str),
        category_summary["Percentage"],
    )

    plt.title(
        "Historical AQI Category Distribution",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("AQI Category")
    plt.ylabel("Percentage of Observations")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.3,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plot_path = (
        RESULTS_DIR
        / "aqi_category_distribution.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nAQI category statistics saved to:"
    )

    print(
        csv_path
    )

    print(
        "\nAQI category plot saved to:"
    )

    print(
        plot_path
    )

def main():
    df = load_data()

    analyze_data_quality(
        df
    )

    plot_aqi_distribution(
        df
    )

    plot_city_aqi_comparison(
        df
    )

    plot_temporal_aqi_trends(
        df
    )

    check_city_series_similarity(
    df
)

    analyze_feature_correlations(
    df
)

    analyze_hourly_aqi_pattern(
    df
)

    analyze_aqi_categories(
    df
)

if __name__ == "__main__":
    main()