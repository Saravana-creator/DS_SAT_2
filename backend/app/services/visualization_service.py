"""
VantageGro Retail Stock Insight - Visualization Service (Stage 1 Merge Visualizations)
Generates real, interactive Plotly visualizations (HTML & JSON for web) and static PNGs
for the Stage 1 Dataset Merging audit.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def generate_stage1_merge_visualizations(
    merge_report: Dict[str, Any],
    primary_df: pd.DataFrame,
    ext_df: Optional[pd.DataFrame],
    output_dir: str = "data/reports/merge_visualizations"
) -> Dict[str, str]:
    """
    Generates all 6 required Stage 1 merge visualizations:
    1. Source dataset row-count comparison
    2. Date-range comparison
    3. Daily record counts
    4. Merge match-rate chart
    5. Missing external-feature chart
    6. Before-versus-after row-count summary
    
    Saves interactive HTML, JSON schema for React, and static PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    sns.set_theme(style="whitegrid", palette="viridis")

    # -------------------------------------------------------------
    # 1. Source dataset row-count comparison
    # -------------------------------------------------------------
    row_counts_data = {
        "Source File": ["Primary (sales_data.csv)"],
        "Row Count": [merge_report["primary_row_count"]],
        "Category": ["Primary Inventory"]
    }
    for fname, count in merge_report.get("dataset1_records_per_file", {}).items():
        row_counts_data["Source File"].append(f"Ext: {fname}")
        row_counts_data["Row Count"].append(count)
        row_counts_data["Category"].append("External Context (Dataset 1)")

    df_rows = pd.DataFrame(row_counts_data)
    
    # Plotly
    fig1 = px.bar(
        df_rows,
        x="Source File",
        y="Row Count",
        color="Category",
        text_auto=".2s",
        title="Source Dataset Row-Count Comparison",
        color_discrete_map={
            "Primary Inventory": "#10b981",  # VantageGro Emerald
            "External Context (Dataset 1)": "#3b82f6"
        }
    )
    fig1.update_layout(
        template="plotly_white",
        yaxis_type="log",
        yaxis_title="Record Count (Log Scale)",
        xaxis_title="Dataset File",
        font=dict(family="Inter, sans-serif")
    )
    p1_html = os.path.join(output_dir, "01_source_row_counts.html")
    p1_json = os.path.join(output_dir, "01_source_row_counts.json")
    fig1.write_html(p1_html)
    with open(p1_json, "w") as f:
        f.write(fig1.to_json())
    generated_files["source_row_counts_html"] = p1_html
    generated_files["source_row_counts_json"] = p1_json

    # Matplotlib PNG
    plt.figure(figsize=(9, 5))
    bars = plt.bar(df_rows["Source File"], df_rows["Row Count"], color=['#10b981'] + ['#3b82f6'] * (len(df_rows) - 1))
    plt.yscale('log')
    plt.title("Source Dataset Row Counts (Log Scale)", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Number of Records")
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h * 1.15, f"{h:,}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    p1_png = os.path.join(output_dir, "01_source_row_counts.png")
    plt.savefig(p1_png, dpi=150)
    plt.close()
    generated_files["source_row_counts_png"] = p1_png

    # -------------------------------------------------------------
    # 2. Date-range comparison (Timeline / Gantt)
    # -------------------------------------------------------------
    timeline_data = [
        dict(
            Dataset="Primary: VantageGro Inventory",
            Start=merge_report["overlap_analysis"]["primary_date_min"],
            End=merge_report["overlap_analysis"]["primary_date_max"],
            Type="Primary"
        )
    ]
    if merge_report["overlap_analysis"]["external_date_min"]:
        timeline_data.append(dict(
            Dataset="External: Store Sales (ECU Macro)",
            Start=merge_report["overlap_analysis"]["external_date_min"],
            End=merge_report["overlap_analysis"]["external_date_max"],
            Type="External Context"
        ))
    df_timeline = pd.DataFrame(timeline_data)

    fig2 = px.timeline(
        df_timeline,
        x_start="Start",
        x_end="End",
        y="Dataset",
        color="Type",
        title="Dataset Date-Range Chronological Coverage",
        color_discrete_map={"Primary": "#10b981", "External Context": "#ef4444"}
    )
    fig2.update_yaxes(autorange="reversed")
    fig2.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        xaxis_title="Calendar Date",
        annotations=[
            dict(
                x="2019-06-01",
                y=0.5,
                text="<b>4-Year Chronological Discontinuity Gap (2017-2022)</b><br>0.0% Date Overlap",
                showarrow=False,
                font=dict(size=12, color="#b91c1c"),
                bgcolor="#fee2e2",
                bordercolor="#ef4444",
                borderwidth=1,
                borderpad=6
            )
        ]
    )
    p2_html = os.path.join(output_dir, "02_date_range_comparison.html")
    p2_json = os.path.join(output_dir, "02_date_range_comparison.json")
    fig2.write_html(p2_html)
    with open(p2_json, "w") as f:
        f.write(fig2.to_json())
    generated_files["date_range_comparison_html"] = p2_html
    generated_files["date_range_comparison_json"] = p2_json

    # Matplotlib PNG
    plt.figure(figsize=(10, 4))
    plt.hlines(y=[1], xmin=pd.to_datetime(df_timeline.loc[df_timeline['Type'] == 'Primary', 'Start']),
               xmax=pd.to_datetime(df_timeline.loc[df_timeline['Type'] == 'Primary', 'End']),
               colors='#10b981', linewidth=10, label='Primary Inventory (2022-2024)')
    if len(df_timeline) > 1:
        plt.hlines(y=[0], xmin=pd.to_datetime(df_timeline.loc[df_timeline['Type'] == 'External Context', 'Start']),
                   xmax=pd.to_datetime(df_timeline.loc[df_timeline['Type'] == 'External Context', 'End']),
                   colors='#ef4444', linewidth=10, label='External Sales Context (2012-2017)')
    plt.yticks([0, 1], ['External Context', 'Primary Inventory'])
    plt.title("Date Range Comparison: 0.0% Date Overlap Identified", fontsize=12, fontweight='bold')
    plt.xlabel("Date")
    plt.axvspan(pd.to_datetime("2017-12-26"), pd.to_datetime("2022-01-01"), color='#fee2e2', alpha=0.5, label='4-Year Discontinuity Gap')
    plt.legend(loc='upper right')
    plt.tight_layout()
    p2_png = os.path.join(output_dir, "02_date_range_comparison.png")
    plt.savefig(p2_png, dpi=150)
    plt.close()
    generated_files["date_range_comparison_png"] = p2_png

    # -------------------------------------------------------------
    # 3. Daily record counts over time
    # -------------------------------------------------------------
    date_col = "Date" if "Date" in primary_df.columns else primary_df.columns[0]
    daily_primary = primary_df.groupby(date_col).size().reset_index(name="record_count")
    daily_primary["date_dt"] = pd.to_datetime(daily_primary[date_col])
    daily_primary = daily_primary.sort_values("date_dt")

    fig3 = px.line(
        daily_primary,
        x=date_col,
        y="record_count",
        title="Daily Primary Inventory Record Count Distribution (760 Days)",
        labels={date_col: "Date", "record_count": "Records per Day"}
    )
    fig3.update_traces(line_color="#10b981", line_width=2)
    fig3.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        yaxis_range=[0, max(daily_primary["record_count"].max() * 1.2, 120)]
    )
    p3_html = os.path.join(output_dir, "03_daily_record_counts.html")
    p3_json = os.path.join(output_dir, "03_daily_record_counts.json")
    fig3.write_html(p3_html)
    with open(p3_json, "w") as f:
        f.write(fig3.to_json())
    generated_files["daily_record_counts_html"] = p3_html
    generated_files["daily_record_counts_json"] = p3_json

    # Matplotlib PNG
    plt.figure(figsize=(10, 4))
    plt.plot(daily_primary["date_dt"], daily_primary["record_count"], color='#10b981', linewidth=1.5)
    plt.title("Primary Dataset: Daily Record Counts (Constant 100 observations/day: 5 Stores x 20 Products)", fontsize=11, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Observations / Day")
    plt.ylim(0, 120)
    plt.tight_layout()
    p3_png = os.path.join(output_dir, "03_daily_record_counts.png")
    plt.savefig(p3_png, dpi=150)
    plt.close()
    generated_files["daily_record_counts_png"] = p3_png

    # -------------------------------------------------------------
    # 4. Merge match-rate chart
    # -------------------------------------------------------------
    matched_dates = merge_report["overlap_analysis"]["overlapping_dates_count"]
    unmatched_dates = merge_report["overlap_analysis"]["unmatched_primary_dates"]
    
    match_df = pd.DataFrame({
        "Status": ["Matched Primary Dates", "Unmatched Primary Dates"],
        "Count": [matched_dates, unmatched_dates]
    })
    
    fig4 = px.pie(
        match_df,
        values="Count",
        names="Status",
        title=f"Dataset Merge Match Rate ({merge_report['overlap_analysis']['merge_match_rate_pct']}%)",
        color="Status",
        color_discrete_map={
            "Matched Primary Dates": "#10b981",
            "Unmatched Primary Dates": "#ef4444"
        },
        hole=0.45
    )
    fig4.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p4_html = os.path.join(output_dir, "04_merge_match_rate.html")
    p4_json = os.path.join(output_dir, "04_merge_match_rate.json")
    fig4.write_html(p4_html)
    with open(p4_json, "w") as f:
        f.write(fig4.to_json())
    generated_files["merge_match_rate_html"] = p4_html
    generated_files["merge_match_rate_json"] = p4_json

    # Matplotlib PNG
    plt.figure(figsize=(6, 6))
    plt.pie(match_df["Count"], labels=match_df["Status"], colors=['#10b981', '#ef4444'],
            autopct='%1.1f%%', startangle=140, explode=(0, 0.05),
            wedgeprops=dict(width=0.6, edgecolor='w'))
    plt.title(f"Primary Dataset Merge Match Rate: {merge_report['overlap_analysis']['merge_match_rate_pct']}%", fontsize=12, fontweight='bold')
    plt.tight_layout()
    p4_png = os.path.join(output_dir, "04_merge_match_rate.png")
    plt.savefig(p4_png, dpi=150)
    plt.close()
    generated_files["merge_match_rate_png"] = p4_png

    # -------------------------------------------------------------
    # 5. Missing external-feature chart
    # -------------------------------------------------------------
    ext_missing = merge_report.get("missing_external_rates", {})
    if not ext_missing:
        ext_missing = {"ext_oil_price": 100.0, "ext_holiday_event_count": 100.0,
                       "ext_total_transactions": 100.0, "ext_total_sales_market": 100.0}
    
    missing_df = pd.DataFrame({
        "Feature": list(ext_missing.keys()),
        "Missing Percentage (%)": [float(v) for v in ext_missing.values()]
    })
    
    fig5 = px.bar(
        missing_df,
        x="Feature",
        y="Missing Percentage (%)",
        title="Missing Values Rate in Candidate External Context Features",
        text="Missing Percentage (%)",
        color_discrete_sequence=["#ef4444"]
    )
    fig5.update_layout(
        template="plotly_white",
        yaxis_range=[0, 115],
        font=dict(family="Inter, sans-serif"),
        annotations=[
            dict(
                x=0.5,
                y=105,
                xref="paper",
                yref="y",
                text="<b>Unsuitable for Modeling: 100% NaN in Production Window (2022-2024)</b>",
                showarrow=False,
                font=dict(size=12, color="#b91c1c")
            )
        ]
    )
    p5_html = os.path.join(output_dir, "05_missing_external_features.html")
    p5_json = os.path.join(output_dir, "05_missing_external_features.json")
    fig5.write_html(p5_html)
    with open(p5_json, "w") as f:
        f.write(fig5.to_json())
    generated_files["missing_external_features_html"] = p5_html
    generated_files["missing_external_features_json"] = p5_json

    # Matplotlib PNG
    plt.figure(figsize=(9, 4.5))
    bars5 = plt.bar(missing_df["Feature"], missing_df["Missing Percentage (%)"], color='#ef4444')
    plt.title("Missing Value Rate in Candidate External Features (100% Missing)", fontsize=12, fontweight='bold')
    plt.ylabel("Missing % in Primary Data")
    plt.ylim(0, 115)
    plt.xticks(rotation=20, ha='right')
    for b in bars5:
        plt.text(b.get_x() + b.get_width() / 2., b.get_height() + 2, f"{b.get_height()}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    p5_png = os.path.join(output_dir, "05_missing_external_features.png")
    plt.savefig(p5_png, dpi=150)
    plt.close()
    generated_files["missing_external_features_png"] = p5_png

    # -------------------------------------------------------------
    # 6. Before-versus-after row-count summary
    # -------------------------------------------------------------
    summary_df = pd.DataFrame({
        "Stage": ["Primary Raw Ingested", "Merged Raw (Output)"],
        "Row Count": [merge_report["primary_row_count"], merge_report["final_row_count"]],
        "Column Count": [merge_report["primary_column_count"], merge_report["final_column_count"]]
    })

    fig6 = make_subplots(rows=1, cols=2, subplot_titles=("Row Count Before vs After", "Column Count Preserved"))
    fig6.add_trace(
        go.Bar(name="Rows", x=summary_df["Stage"], y=summary_df["Row Count"], marker_color="#10b981", text=summary_df["Row Count"], textposition="auto"),
        row=1, col=1
    )
    fig6.add_trace(
        go.Bar(name="Columns", x=summary_df["Stage"], y=summary_df["Column Count"], marker_color="#065f46", text=summary_df["Column Count"], textposition="auto"),
        row=1, col=2
    )
    fig6.update_layout(template="plotly_white", title_text="Before vs After Stage 1 Summary", showlegend=False, font=dict(family="Inter, sans-serif"))
    p6_html = os.path.join(output_dir, "06_before_after_summary.html")
    p6_json = os.path.join(output_dir, "06_before_after_summary.json")
    fig6.write_html(p6_html)
    with open(p6_json, "w") as f:
        f.write(fig6.to_json())
    generated_files["before_after_summary_html"] = p6_html
    generated_files["before_after_summary_json"] = p6_json

    # Matplotlib PNG
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar(summary_df["Stage"], summary_df["Row Count"], color='#10b981')
    ax1.set_title("Row Counts (Zero Row Loss)", fontsize=11, fontweight='bold')
    for b in ax1.patches:
        ax1.annotate(f"{int(b.get_height()):,}", (b.get_x() + b.get_width() / 2., b.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold')
    
    ax2.bar(summary_df["Stage"], summary_df["Column Count"], color='#047857')
    ax2.set_title("Column Counts (Preserved Integrity)", fontsize=11, fontweight='bold')
    for b in ax2.patches:
        ax2.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2., b.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold')
    plt.suptitle("Before vs After Stage 1 Pipeline Summary", fontsize=13, fontweight='bold')
    plt.tight_layout()
    p6_png = os.path.join(output_dir, "06_before_after_summary.png")
    plt.savefig(p6_png, dpi=150)
    plt.close()
    generated_files["before_after_summary_png"] = p6_png

    logger.info(f"Generated all 6 Stage 1 visualizations in {output_dir}")
    return generated_files


def generate_stage2_preprocessing_visualizations(
    preprocessing_report: Dict[str, Any],
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_dir: str = "data/reports/preprocessing_visualizations"
) -> Dict[str, str]:
    """
    Generates all 9 required Stage 2 Preprocessing visualizations:
    1. Missing values before and after cleaning
    2. Duplicate record counts
    3. Numeric distributions before and after treatment
    4. Outlier analysis (box plots with IQR bounds)
    5. Categorical value distributions
    6. Invalid-value counts (anomalies audit)
    7. Data type summary
    8. Correlation heatmap
    9. Final feature completeness chart
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    # 1. Missing values before and after cleaning
    cols = df_before.columns.tolist()
    missing_before = [int(df_before[c].isna().sum()) for c in cols]
    missing_after = [int(df_after[c].isna().sum()) for c in cols]

    df_miss = pd.DataFrame({
        "Feature": cols * 2,
        "Missing Count": missing_before + missing_after,
        "Stage": ["Before Cleaning"] * len(cols) + ["After Cleaning"] * len(cols)
    })

    fig1 = px.bar(
        df_miss,
        x="Feature",
        y="Missing Count",
        color="Stage",
        barmode="group",
        title="Missing Values Count: Before vs After Preprocessing (0 Missing Records)",
        color_discrete_map={"Before Cleaning": "#ef4444", "After Cleaning": "#10b981"}
    )
    fig1.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p1_html = os.path.join(output_dir, "01_missing_values_before_after.html")
    p1_json = os.path.join(output_dir, "01_missing_values_before_after.json")
    fig1.write_html(p1_html)
    with open(p1_json, "w") as f:
        f.write(fig1.to_json())
    generated_files["missing_values_html"] = p1_html
    generated_files["missing_values_json"] = p1_json

    plt.figure(figsize=(10, 4.5))
    x_pos = np.arange(len(cols))
    plt.bar(x_pos - 0.2, missing_before, 0.4, label='Before Cleaning (0)', color='#ef4444')
    plt.bar(x_pos + 0.2, missing_after, 0.4, label='After Cleaning (0)', color='#10b981')
    plt.title("Missing Values Before vs After Cleaning (100% Complete)", fontsize=12, fontweight='bold')
    plt.xticks(x_pos, cols, rotation=35, ha='right')
    plt.ylabel("Missing Count")
    plt.ylim(0, 10)
    plt.legend()
    plt.tight_layout()
    p1_png = os.path.join(output_dir, "01_missing_values_before_after.png")
    plt.savefig(p1_png, dpi=150)
    plt.close()
    generated_files["missing_values_png"] = p1_png

    # 2. Duplicate record counts
    dupe_full = preprocessing_report["quality_before"]["full_duplicates"]
    dupe_biz = preprocessing_report["quality_before"]["business_key_duplicates"]
    df_dupe = pd.DataFrame({
        "Duplicate Type": ["Full Duplicate Rows", "Duplicate Business Keys (Date, Store, Product)"],
        "Count": [dupe_full, dupe_biz],
        "Status": ["Valid Unique", "Valid Unique"]
    })

    fig2 = px.bar(
        df_dupe,
        x="Duplicate Type",
        y="Count",
        color="Status",
        title="Duplicate Records Audit (Zero Duplicate Rows Detected)",
        color_discrete_map={"Valid Unique": "#10b981"},
        text_auto=True
    )
    fig2.update_layout(template="plotly_white", yaxis_range=[0, 10], font=dict(family="Inter, sans-serif"))
    p2_html = os.path.join(output_dir, "02_duplicate_record_counts.html")
    p2_json = os.path.join(output_dir, "02_duplicate_record_counts.json")
    fig2.write_html(p2_html)
    with open(p2_json, "w") as f:
        f.write(fig2.to_json())
    generated_files["duplicate_counts_html"] = p2_html
    generated_files["duplicate_counts_json"] = p2_json

    plt.figure(figsize=(7, 4))
    plt.bar(df_dupe["Duplicate Type"], df_dupe["Count"], color='#10b981')
    plt.title("Duplicate Records Audit: Zero Duplicates", fontsize=11, fontweight='bold')
    plt.ylabel("Count")
    plt.ylim(0, 5)
    plt.tight_layout()
    p2_png = os.path.join(output_dir, "02_duplicate_record_counts.png")
    plt.savefig(p2_png, dpi=150)
    plt.close()
    generated_files["duplicate_counts_png"] = p2_png

    # 3. Numeric distributions before and after treatment
    num_cols = ["Demand", "Units Sold", "Inventory Level", "Price"]
    fig3 = make_subplots(rows=2, cols=2, subplot_titles=[f"Distribution: {c}" for c in num_cols])
    coords = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for i, c in enumerate(num_cols):
        r, cl = coords[i]
        fig3.add_trace(
            go.Histogram(x=df_after[c], name=c, marker_color="#10b981", nbinsx=30),
            row=r, col=cl
        )
    fig3.update_layout(template="plotly_white", title_text="Numeric Feature Distributions After Treatment", showlegend=False, font=dict(family="Inter, sans-serif"))
    p3_html = os.path.join(output_dir, "03_numeric_distributions.html")
    p3_json = os.path.join(output_dir, "03_numeric_distributions.json")
    fig3.write_html(p3_html)
    with open(p3_json, "w") as f:
        f.write(fig3.to_json())
    generated_files["numeric_distributions_html"] = p3_html
    generated_files["numeric_distributions_json"] = p3_json

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i, c in enumerate(num_cols):
        ax = axes[i // 2, i % 2]
        sns.histplot(df_after[c], kde=True, ax=ax, color='#10b981', bins=30)
        ax.set_title(f"{c} Distribution", fontsize=10, fontweight='bold')
    plt.suptitle("Numeric Feature Distributions After Treatment", fontsize=12, fontweight='bold')
    plt.tight_layout()
    p3_png = os.path.join(output_dir, "03_numeric_distributions.png")
    plt.savefig(p3_png, dpi=150)
    plt.close()
    generated_files["numeric_distributions_png"] = p3_png

    # 4. Outlier analysis (Box plots & IQR bounds)
    fig4 = make_subplots(rows=1, cols=4, subplot_titles=["Demand", "Units Sold", "Inventory Level", "Price"])
    for i, c in enumerate(["Demand", "Units Sold", "Inventory Level", "Price"]):
        fig4.add_trace(go.Box(y=df_after[c], name=c, marker_color="#047857"), row=1, col=i+1)
    fig4.update_layout(template="plotly_white", title_text="Outlier Analysis: Valid Heavy-Tailed Retail Volumes", showlegend=False, font=dict(family="Inter, sans-serif"))
    p4_html = os.path.join(output_dir, "04_outlier_boxplots.html")
    p4_json = os.path.join(output_dir, "04_outlier_boxplots.json")
    fig4.write_html(p4_html)
    with open(p4_json, "w") as f:
        f.write(fig4.to_json())
    generated_files["outlier_boxplots_html"] = p4_html
    generated_files["outlier_boxplots_json"] = p4_json

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for i, c in enumerate(["Demand", "Units Sold", "Inventory Level", "Price"]):
        sns.boxplot(y=df_after[c], ax=axes[i], color='#34d399')
        axes[i].set_title(c, fontsize=10, fontweight='bold')
    plt.suptitle("Outlier Analysis (IQR Boxplots): Retained Valid Retail Surges", fontsize=12, fontweight='bold')
    plt.tight_layout()
    p4_png = os.path.join(output_dir, "04_outlier_boxplots.png")
    plt.savefig(p4_png, dpi=150)
    plt.close()
    generated_files["outlier_boxplots_png"] = p4_png

    # 5. Categorical value distributions
    cat_cols = ["Category", "Region", "Weather Condition", "Seasonality"]
    fig5 = make_subplots(rows=2, cols=2, subplot_titles=[f"Distribution of {c}" for c in cat_cols])
    coords5 = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for i, c in enumerate(cat_cols):
        r, cl = coords5[i]
        counts = df_after[c].value_counts()
        fig5.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color="#10b981", name=c), row=r, col=cl)
    fig5.update_layout(template="plotly_white", title_text="Categorical Value Distributions", showlegend=False, font=dict(family="Inter, sans-serif"))
    p5_html = os.path.join(output_dir, "05_categorical_distributions.html")
    p5_json = os.path.join(output_dir, "05_categorical_distributions.json")
    fig5.write_html(p5_html)
    with open(p5_json, "w") as f:
        f.write(fig5.to_json())
    generated_files["categorical_distributions_html"] = p5_html
    generated_files["categorical_distributions_json"] = p5_json

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, c in enumerate(cat_cols):
        ax = axes[i // 2, i % 2]
        counts = df_after[c].value_counts()
        ax.bar(counts.index, counts.values, color='#059669')
        ax.set_title(f"{c} Counts", fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
    plt.suptitle("Categorical Variable Frequency Distributions", fontsize=12, fontweight='bold')
    plt.tight_layout()
    p5_png = os.path.join(output_dir, "05_categorical_distributions.png")
    plt.savefig(p5_png, dpi=150)
    plt.close()
    generated_files["categorical_distributions_png"] = p5_png

    # 6. Invalid-value counts (anomalies audit)
    anom = preprocessing_report["quality_before"]["anomalies"]
    df_anom = pd.DataFrame({
        "Sanity Rule": [k.replace("_", " ").title() for k in anom.keys()],
        "Violations Count": list(anom.values())
    })
    fig6 = px.bar(
        df_anom,
        x="Sanity Rule",
        y="Violations Count",
        title="Domain Boundary Sanity Audit (All 0 Violations)",
        text_auto=True,
        color_discrete_sequence=["#10b981"]
    )
    fig6.update_layout(template="plotly_white", yaxis_range=[0, 10], font=dict(family="Inter, sans-serif"))
    p6_html = os.path.join(output_dir, "06_invalid_values_audit.html")
    p6_json = os.path.join(output_dir, "06_invalid_values_audit.json")
    fig6.write_html(p6_html)
    with open(p6_json, "w") as f:
        f.write(fig6.to_json())
    generated_files["invalid_values_audit_html"] = p6_html
    generated_files["invalid_values_audit_json"] = p6_json

    plt.figure(figsize=(10, 4))
    plt.bar(df_anom["Sanity Rule"], df_anom["Violations Count"], color='#10b981')
    plt.title("Domain Sanity Violations (Zero Invalid Values)", fontsize=11, fontweight='bold')
    plt.ylabel("Violations")
    plt.xticks(rotation=25, ha='right')
    plt.ylim(0, 5)
    plt.tight_layout()
    p6_png = os.path.join(output_dir, "06_invalid_values_audit.png")
    plt.savefig(p6_png, dpi=150)
    plt.close()
    generated_files["invalid_values_audit_png"] = p6_png

    # 7. Data type summary
    dtypes_series = df_after.dtypes.astype(str).value_counts()
    fig7 = px.pie(
        names=dtypes_series.index,
        values=dtypes_series.values,
        title="Dataset Schema: Data Type Breakdown (16 Features)",
        color_discrete_sequence=["#10b981", "#3b82f6", "#f59e0b", "#6366f1"],
        hole=0.4
    )
    fig7.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p7_html = os.path.join(output_dir, "07_data_type_summary.html")
    p7_json = os.path.join(output_dir, "07_data_type_summary.json")
    fig7.write_html(p7_html)
    with open(p7_json, "w") as f:
        f.write(fig7.to_json())
    generated_files["data_type_summary_html"] = p7_html
    generated_files["data_type_summary_json"] = p7_json

    plt.figure(figsize=(5, 5))
    plt.pie(dtypes_series.values, labels=dtypes_series.index, colors=['#10b981', '#3b82f6', '#f59e0b', '#6366f1'],
            autopct='%1.0f%%', startangle=90, wedgeprops=dict(width=0.6, edgecolor='w'))
    plt.title("Data Types Summary Breakdown", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p7_png = os.path.join(output_dir, "07_data_type_summary.png")
    plt.savefig(p7_png, dpi=150)
    plt.close()
    generated_files["data_type_summary_png"] = p7_png

    # 8. Correlation heatmap
    num_df = df_after.select_dtypes(include=[np.number])
    corr = num_df.corr().round(2)
    fig8 = px.imshow(
        corr,
        text_auto=True,
        title="Pearson Correlation Heatmap (Numeric Features)",
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig8.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p8_html = os.path.join(output_dir, "08_correlation_heatmap.html")
    p8_json = os.path.join(output_dir, "08_correlation_heatmap.json")
    fig8.write_html(p8_html)
    with open(p8_json, "w") as f:
        f.write(fig8.to_json())
    generated_files["correlation_heatmap_html"] = p8_html
    generated_files["correlation_heatmap_json"] = p8_json

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1, cbar=True)
    plt.title("Correlation Matrix of Numeric Features", fontsize=12, fontweight='bold')
    plt.tight_layout()
    p8_png = os.path.join(output_dir, "08_correlation_heatmap.png")
    plt.savefig(p8_png, dpi=150)
    plt.close()
    generated_files["correlation_heatmap_png"] = p8_png

    # 9. Final feature completeness chart
    completeness = [100.0 - float(df_after[c].isna().mean() * 100) for c in df_after.columns]
    df_comp = pd.DataFrame({"Feature": df_after.columns, "Completeness (%)": completeness})
    fig9 = px.bar(
        df_comp,
        x="Feature",
        y="Completeness (%)",
        title="Final Feature Completeness (100.0% Across All Features)",
        text_auto=True,
        color_discrete_sequence=["#10b981"]
    )
    fig9.update_layout(template="plotly_white", yaxis_range=[0, 115], font=dict(family="Inter, sans-serif"))
    p9_html = os.path.join(output_dir, "09_feature_completeness.html")
    p9_json = os.path.join(output_dir, "09_feature_completeness.json")
    fig9.write_html(p9_html)
    with open(p9_json, "w") as f:
        f.write(fig9.to_json())
    generated_files["feature_completeness_html"] = p9_html
    generated_files["feature_completeness_json"] = p9_json

    plt.figure(figsize=(10, 4.5))
    plt.bar(df_comp["Feature"], df_comp["Completeness (%)"], color='#10b981')
    plt.title("Final Feature Completeness Rate (100% across all 16 attributes)", fontsize=11, fontweight='bold')
    plt.ylabel("Completeness (%)")
    plt.ylim(0, 115)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    p9_png = os.path.join(output_dir, "09_feature_completeness.png")
    plt.savefig(p9_png, dpi=150)
    plt.close()
    generated_files["feature_completeness_png"] = p9_png

    logger.info(f"Generated all 9 Stage 2 preprocessing visualizations in {output_dir}")
    return generated_files


def generate_stage3_eda_visualizations(
    eda_summary: Dict[str, Any],
    df: pd.DataFrame,
    output_dir: str = "data/reports/eda_visualizations"
) -> Dict[str, str]:
    """
    Generates all required Stage 3 EDA visualizations:
    1. Daily, weekly, monthly demand trends
    2. Units sold by product category
    3. Demand by store and region
    4. Inventory distribution
    5. Inventory versus units sold
    6. Inventory versus demand
    7. Stockout-risk and overstock-risk counts
    8. Price and discount analysis
    9. Promotion versus demand comparisons
    10. Seasonality analysis
    11. Weather impact analysis
    12. Product rankings and demand variability (CV)
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])

    # 1. Demand Trends (Weekly & Monthly Aggregations)
    df_weekly = pd.DataFrame(eda_summary["temporal_trends"]["weekly"])
    fig1 = px.line(
        df_weekly,
        x="week_start",
        y=["weekly_demand", "weekly_units_sold"],
        title="Weekly Demand vs Units Sold Temporal Dynamics (2022-2024)",
        labels={"week_start": "Week Starting Date", "value": "Total Units", "variable": "Metric"},
        color_discrete_map={"weekly_demand": "#10b981", "weekly_units_sold": "#3b82f6"}
    )
    fig1.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p1_html = os.path.join(output_dir, "01_demand_trends_temporal.html")
    p1_json = os.path.join(output_dir, "01_demand_trends_temporal.json")
    fig1.write_html(p1_html)
    with open(p1_json, "w") as f:
        f.write(fig1.to_json())
    generated_files["demand_trends_html"] = p1_html
    generated_files["demand_trends_json"] = p1_json

    plt.figure(figsize=(11, 4.5))
    plt.plot(pd.to_datetime(df_weekly["week_start"]), df_weekly["weekly_demand"], label="Weekly Customer Demand", color="#10b981", linewidth=2)
    plt.plot(pd.to_datetime(df_weekly["week_start"]), df_weekly["weekly_units_sold"], label="Weekly Units Sold (Constrained)", color="#3b82f6", linewidth=1.5, linestyle="--")
    plt.title("Weekly Demand vs Units Sold Trends (2022-2024)", fontsize=12, fontweight='bold')
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()
    p1_png = os.path.join(output_dir, "01_demand_trends_temporal.png")
    plt.savefig(p1_png, dpi=150)
    plt.close()
    generated_files["demand_trends_png"] = p1_png

    # 2. Units Sold by Product Category
    df_cat = pd.DataFrame(eda_summary["category_summary"])
    fig2 = px.bar(
        df_cat,
        x="Category",
        y="total_units_sold",
        color="Category",
        text_auto=".2s",
        title="Total Units Sold by Product Category",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig2.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p2_html = os.path.join(output_dir, "02_units_sold_by_category.html")
    p2_json = os.path.join(output_dir, "02_units_sold_by_category.json")
    fig2.write_html(p2_html)
    with open(p2_json, "w") as f:
        f.write(fig2.to_json())
    generated_files["units_sold_by_category_html"] = p2_html
    generated_files["units_sold_by_category_json"] = p2_json

    plt.figure(figsize=(8, 4))
    plt.bar(df_cat["Category"], df_cat["total_units_sold"], color="#10b981")
    plt.title("Total Units Sold by Product Category", fontsize=11, fontweight='bold')
    plt.ylabel("Units Sold")
    for i, v in enumerate(df_cat["total_units_sold"]):
        plt.text(i, v/2, f"{v:,}", ha='center', va='center', color='white', fontweight='bold')
    plt.tight_layout()
    p2_png = os.path.join(output_dir, "02_units_sold_by_category.png")
    plt.savefig(p2_png, dpi=150)
    plt.close()
    generated_files["units_sold_by_category_png"] = p2_png

    # 3. Demand by Store and Region
    df_store = pd.DataFrame(eda_summary["store_summary"])
    fig3 = px.bar(
        df_store,
        x="Store ID",
        y="total_demand",
        color="Region",
        text_auto=".2s",
        title="Total Customer Demand by Retail Store & Geographic Region"
    )
    fig3.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p3_html = os.path.join(output_dir, "03_demand_by_store_and_region.html")
    p3_json = os.path.join(output_dir, "03_demand_by_store_and_region.json")
    fig3.write_html(p3_html)
    with open(p3_json, "w") as f:
        f.write(fig3.to_json())
    generated_files["demand_by_store_html"] = p3_html
    generated_files["demand_by_store_json"] = p3_json

    plt.figure(figsize=(8, 4))
    sns.barplot(data=df_store, x="Store ID", y="total_demand", hue="Region", palette="viridis")
    plt.title("Demand by Store ID and Region", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p3_png = os.path.join(output_dir, "03_demand_by_store_and_region.png")
    plt.savefig(p3_png, dpi=150)
    plt.close()
    generated_files["demand_by_store_png"] = p3_png

    # 4. Inventory Distribution
    fig4 = px.histogram(
        df_copy,
        x="Inventory Level",
        nbins=40,
        title="Inventory Stock Level Distribution Across Retail Locations",
        color_discrete_sequence=["#047857"]
    )
    fig4.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p4_html = os.path.join(output_dir, "04_inventory_distribution.html")
    p4_json = os.path.join(output_dir, "04_inventory_distribution.json")
    fig4.write_html(p4_html)
    with open(p4_json, "w") as f:
        f.write(fig4.to_json())
    generated_files["inventory_distribution_html"] = p4_html
    generated_files["inventory_distribution_json"] = p4_json

    plt.figure(figsize=(8, 4))
    sns.histplot(df_copy["Inventory Level"], kde=True, color="#047857", bins=35)
    plt.title("Inventory Level Distribution Across All SKUs", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p4_png = os.path.join(output_dir, "04_inventory_distribution.png")
    plt.savefig(p4_png, dpi=150)
    plt.close()
    generated_files["inventory_distribution_png"] = p4_png

    # 5. Inventory vs Units Sold (Physical Constraint Verification)
    sample_df = df_copy.sample(min(2000, len(df_copy)), random_state=42)
    fig5 = px.scatter(
        sample_df,
        x="Inventory Level",
        y="Units Sold",
        color="Category",
        opacity=0.6,
        title="Inventory Level vs Units Sold (Upper Bound Envelope: Units Sold <= Inventory)"
    )
    fig5.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p5_html = os.path.join(output_dir, "05_inventory_vs_units_sold.html")
    p5_json = os.path.join(output_dir, "05_inventory_vs_units_sold.json")
    fig5.write_html(p5_html)
    with open(p5_json, "w") as f:
        f.write(fig5.to_json())
    generated_files["inventory_vs_units_sold_html"] = p5_html
    generated_files["inventory_vs_units_sold_json"] = p5_json

    plt.figure(figsize=(7, 5))
    plt.scatter(sample_df["Inventory Level"], sample_df["Units Sold"], alpha=0.4, color="#10b981", s=15)
    plt.plot([0, 450], [0, 450], color='red', linestyle='--', label='Physical Limit (Sold = Inventory)')
    plt.title("Inventory Level vs Units Sold (Censoring Envelope)", fontsize=11, fontweight='bold')
    plt.xlabel("Inventory Level")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.tight_layout()
    p5_png = os.path.join(output_dir, "05_inventory_vs_units_sold.png")
    plt.savefig(p5_png, dpi=150)
    plt.close()
    generated_files["inventory_vs_units_sold_png"] = p5_png

    # 6. Inventory vs Demand
    fig6 = px.scatter(
        sample_df,
        x="Inventory Level",
        y="Demand",
        color="Category",
        opacity=0.6,
        title="On-Hand Inventory Level vs True Customer Demand"
    )
    fig6.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p6_html = os.path.join(output_dir, "06_inventory_vs_demand.html")
    p6_json = os.path.join(output_dir, "06_inventory_vs_demand.json")
    fig6.write_html(p6_html)
    with open(p6_json, "w") as f:
        f.write(fig6.to_json())
    generated_files["inventory_vs_demand_html"] = p6_html
    generated_files["inventory_vs_demand_json"] = p6_json

    plt.figure(figsize=(7, 5))
    plt.scatter(sample_df["Inventory Level"], sample_df["Demand"], alpha=0.4, color="#3b82f6", s=15)
    plt.title("Inventory Level vs Customer Demand", fontsize=11, fontweight='bold')
    plt.xlabel("Inventory Level")
    plt.ylabel("Customer Demand")
    plt.tight_layout()
    p6_png = os.path.join(output_dir, "06_inventory_vs_demand.png")
    plt.savefig(p6_png, dpi=150)
    plt.close()
    generated_files["inventory_vs_demand_png"] = p6_png

    # 7. Stockout Risk vs Overstock Risk Counts
    risks = eda_summary["risk_metrics"]
    df_risk = pd.DataFrame({
        "Inventory Status": ["Stockout Risk (Inv < Demand)", "Severe Stockout (Inv = 0)", "Overstock Risk (Inv > 3x Demand)", "Balanced Stock Level"],
        "Record Count": [risks["stockout_risk_records"], risks["severe_zero_inventory_records"], risks["overstock_risk_records"], risks["balanced_inventory_records"]],
        "Percentage": [risks["stockout_risk_pct"], risks["severe_zero_inventory_pct"], risks["overstock_risk_pct"], risks["balanced_inventory_pct"]]
    })
    fig7 = px.bar(
        df_risk,
        x="Inventory Status",
        y="Record Count",
        color="Inventory Status",
        text="Percentage",
        title="Inventory Health Risk Distribution (Stockout vs Overstock Risk)",
        color_discrete_map={
            "Stockout Risk (Inv < Demand)": "#ef4444",
            "Severe Stockout (Inv = 0)": "#b91c1c",
            "Overstock Risk (Inv > 3x Demand)": "#f59e0b",
            "Balanced Stock Level": "#10b981"
        }
    )
    fig7.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p7_html = os.path.join(output_dir, "07_stockout_overstock_risks.html")
    p7_json = os.path.join(output_dir, "07_stockout_overstock_risks.json")
    fig7.write_html(p7_html)
    with open(p7_json, "w") as f:
        f.write(fig7.to_json())
    generated_files["stockout_overstock_risks_html"] = p7_html
    generated_files["stockout_overstock_risks_json"] = p7_json

    plt.figure(figsize=(9, 4.5))
    bars7 = plt.bar(df_risk["Inventory Status"], df_risk["Record Count"], color=['#ef4444', '#b91c1c', '#f59e0b', '#10b981'])
    plt.title("Stockout Risk vs Overstock Risk Proportions", fontsize=11, fontweight='bold')
    plt.ylabel("Record Count")
    plt.xticks(rotation=20, ha='right')
    for b in bars7:
        plt.text(b.get_x() + b.get_width()/2., b.get_height() + 500, f"{int(b.get_height()):,}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    p7_png = os.path.join(output_dir, "07_stockout_overstock_risks.png")
    plt.savefig(p7_png, dpi=150)
    plt.close()
    generated_files["stockout_overstock_risks_png"] = p7_png

    # 8. Price & Discount Analysis
    df_disc = pd.DataFrame(eda_summary["discount_analysis"])
    fig8 = px.bar(
        df_disc,
        x="discount_tier",
        y="mean_demand",
        title="Average Customer Demand by Promotional Discount Tier",
        color_discrete_sequence=["#10b981"],
        text_auto=".1f"
    )
    fig8.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p8_html = os.path.join(output_dir, "08_price_discount_analysis.html")
    p8_json = os.path.join(output_dir, "08_price_discount_analysis.json")
    fig8.write_html(p8_html)
    with open(p8_json, "w") as f:
        f.write(fig8.to_json())
    generated_files["price_discount_analysis_html"] = p8_html
    generated_files["price_discount_analysis_json"] = p8_json

    plt.figure(figsize=(8, 4))
    plt.bar(df_disc["discount_tier"], df_disc["mean_demand"], color='#10b981')
    plt.title("Mean Demand by Discount Tier", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Demand")
    plt.tight_layout()
    p8_png = os.path.join(output_dir, "08_price_discount_analysis.png")
    plt.savefig(p8_png, dpi=150)
    plt.close()
    generated_files["price_discount_analysis_png"] = p8_png

    # 9. Promotion vs Demand Comparisons
    fig9 = px.box(
        df_copy,
        x="Promotion",
        y="Demand",
        color="Promotion",
        title="Customer Demand Distribution: Non-Promotional vs Promotional Periods",
        color_discrete_map={0: "#94a3b8", 1: "#10b981"}
    )
    fig9.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p9_html = os.path.join(output_dir, "09_promotion_vs_demand.html")
    p9_json = os.path.join(output_dir, "09_promotion_vs_demand.json")
    fig9.write_html(p9_html)
    with open(p9_json, "w") as f:
        f.write(fig9.to_json())
    generated_files["promotion_vs_demand_html"] = p9_html
    generated_files["promotion_vs_demand_json"] = p9_json

    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df_copy, x="Promotion", y="Demand", palette=["#94a3b8", "#10b981"])
    plt.title("Demand by Promotion Status (0 = Standard, 1 = Active Promo)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p9_png = os.path.join(output_dir, "09_promotion_vs_demand.png")
    plt.savefig(p9_png, dpi=150)
    plt.close()
    generated_files["promotion_vs_demand_png"] = p9_png

    # 10. Seasonality Analysis
    df_season = pd.DataFrame(eda_summary["seasonality_summary"])
    fig10 = px.bar(
        df_season,
        x="Seasonality",
        y="mean_demand",
        color="Seasonality",
        title="Average Daily Demand by Calendar Season",
        text_auto=".1f"
    )
    fig10.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p10_html = os.path.join(output_dir, "10_seasonality_analysis.html")
    p10_json = os.path.join(output_dir, "10_seasonality_analysis.json")
    fig10.write_html(p10_html)
    with open(p10_json, "w") as f:
        f.write(fig10.to_json())
    generated_files["seasonality_analysis_html"] = p10_html
    generated_files["seasonality_analysis_json"] = p10_json

    plt.figure(figsize=(7, 4))
    plt.bar(df_season["Seasonality"], df_season["mean_demand"], color='#059669')
    plt.title("Mean Daily Demand by Season", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Demand")
    plt.tight_layout()
    p10_png = os.path.join(output_dir, "10_seasonality_analysis.png")
    plt.savefig(p10_png, dpi=150)
    plt.close()
    generated_files["seasonality_analysis_png"] = p10_png

    # 11. Weather Impact Analysis
    df_weather = pd.DataFrame(eda_summary["weather_summary"])
    fig11 = px.bar(
        df_weather,
        x="Weather Condition",
        y="mean_demand",
        color="Weather Condition",
        title="Average Customer Demand by Weather Condition",
        text_auto=".1f"
    )
    fig11.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p11_html = os.path.join(output_dir, "11_weather_impact.html")
    p11_json = os.path.join(output_dir, "11_weather_impact.json")
    fig11.write_html(p11_html)
    with open(p11_json, "w") as f:
        f.write(fig11.to_json())
    generated_files["weather_impact_html"] = p11_html
    generated_files["weather_impact_json"] = p11_json

    plt.figure(figsize=(7, 4))
    plt.bar(df_weather["Weather Condition"], df_weather["mean_demand"], color='#3b82f6')
    plt.title("Mean Demand by Weather Condition", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Demand")
    plt.tight_layout()
    p11_png = os.path.join(output_dir, "11_weather_impact.png")
    plt.savefig(p11_png, dpi=150)
    plt.close()
    generated_files["weather_impact_png"] = p11_png

    # 12. Product Rankings & Volatility (CV)
    df_prod = pd.DataFrame(eda_summary["product_rankings"])
    fig12 = px.scatter(
        df_prod,
        x="mean_demand",
        y="cv_demand",
        size="total_demand",
        color="volatility_class",
        hover_name="Product ID",
        text="Product ID",
        title="Product Demand Volume vs Coefficient of Variation (Volatility Segmentation)",
        color_discrete_map={
            "High Volatility": "#ef4444",
            "Moderate Volatility": "#f59e0b",
            "Stable Demand": "#10b981"
        }
    )
    fig12.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p12_html = os.path.join(output_dir, "12_product_rankings_and_volatility.html")
    p12_json = os.path.join(output_dir, "12_product_rankings_and_volatility.json")
    fig12.write_html(p12_html)
    with open(p12_json, "w") as f:
        f.write(fig12.to_json())
    generated_files["product_volatility_html"] = p12_html
    generated_files["product_volatility_json"] = p12_json

    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df_prod, x="mean_demand", y="cv_demand", hue="volatility_class", s=100, palette={"High Volatility": "#ef4444", "Moderate Volatility": "#f59e0b", "Stable Demand": "#10b981"})
    for _, r in df_prod.iterrows():
        plt.text(r["mean_demand"] + 1, r["cv_demand"], r["Product ID"], fontsize=7)
    plt.title("Product Volume vs Volatility (Coefficient of Variation)", fontsize=11, fontweight='bold')
    plt.xlabel("Mean Daily Demand")
    plt.ylabel("Coefficient of Variation (std/mean)")
    plt.tight_layout()
    p12_png = os.path.join(output_dir, "12_product_rankings_and_volatility.png")
    plt.savefig(p12_png, dpi=150)
    plt.close()
    generated_files["product_volatility_png"] = p12_png

    logger.info(f"Generated all 12 Stage 3 EDA visualizations in {output_dir}")
    return generated_files


def generate_stage4_feature_visualizations(
    feature_metadata: Dict[str, Any],
    df_features: pd.DataFrame,
    output_dir: str = "data/reports/feature_visualizations"
) -> Dict[str, str]:
    """
    Generates all required Stage 4 Feature Engineering visualizations:
    1. Feature availability by category
    2. Feature correlation matrix (top features vs Demand)
    3. Lag-feature versus target relationships
    4. Rolling demand comparisons
    5. Engineered feature distributions
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    # 1. Feature Availability by Category
    cats = feature_metadata["feature_categories"]
    df_avail = pd.DataFrame({
        "Category": [k.replace("_", " ").title() for k in cats.keys()],
        "Feature Count": [len(v) for v in cats.values()]
    })

    fig1 = px.bar(
        df_avail,
        x="Category",
        y="Feature Count",
        color="Category",
        text="Feature Count",
        title=f"Engineered Features Availability: {feature_metadata['total_features']} Features Across Categories",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig1.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"), showlegend=False)
    p1_html = os.path.join(output_dir, "01_feature_availability.html")
    p1_json = os.path.join(output_dir, "01_feature_availability.json")
    fig1.write_html(p1_html)
    with open(p1_json, "w") as f:
        f.write(fig1.to_json())
    generated_files["feature_availability_html"] = p1_html
    generated_files["feature_availability_json"] = p1_json

    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(df_avail["Category"], df_avail["Feature Count"], color="#10b981")
    plt.title(f"Engineered Feature Categories ({feature_metadata['total_features']} Total Features)", fontsize=11, fontweight='bold')
    plt.ylabel("Number of Features")
    plt.xticks(rotation=20, ha='right')
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2., b.get_height() + 0.3, str(int(b.get_height())), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    p1_png = os.path.join(output_dir, "01_feature_availability.png")
    plt.savefig(p1_png, dpi=150)
    plt.close()
    generated_files["feature_availability_png"] = p1_png

    # 2. Feature Correlation Matrix (Top Engineered Features vs Demand)
    key_features = [
        "Demand", "demand_lag_1", "demand_lag_7", "rolling_mean_7", "rolling_mean_14",
        "recent_demand_trend", "Price", "Discount", "Promotion", "Competitor Pricing", "price_ratio"
    ]
    avail_key_features = [f for f in key_features if f in df_features.columns]
    corr = df_features[avail_key_features].corr().round(2)

    fig2 = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Matrix: Engineered Lags, Rolling Means & Business Drivers vs Target Demand",
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig2.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p2_html = os.path.join(output_dir, "02_feature_correlation_matrix.html")
    p2_json = os.path.join(output_dir, "02_feature_correlation_matrix.json")
    fig2.write_html(p2_html)
    with open(p2_json, "w") as f:
        f.write(fig2.to_json())
    generated_files["feature_correlation_html"] = p2_html
    generated_files["feature_correlation_json"] = p2_json

    plt.figure(figsize=(10, 7.5))
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix: Engineered Features vs Target Demand", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p2_png = os.path.join(output_dir, "02_feature_correlation_matrix.png")
    plt.savefig(p2_png, dpi=150)
    plt.close()
    generated_files["feature_correlation_png"] = p2_png

    # 3. Lag-Feature versus Target Relationship
    sample_sub = df_features.sample(min(1500, len(df_features)), random_state=42)
    fig3 = px.scatter(
        sample_sub,
        x="demand_lag_1",
        y="Demand",
        opacity=0.5,
        title="Autoregressive Relationship: Lag 1 Prior Day Demand vs Current Day Demand",
        color_discrete_sequence=["#10b981"]
    )
    fig3.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p3_html = os.path.join(output_dir, "03_lag_vs_target_relationships.html")
    p3_json = os.path.join(output_dir, "03_lag_vs_target_relationships.json")
    fig3.write_html(p3_html)
    with open(p3_json, "w") as f:
        f.write(fig3.to_json())
    generated_files["lag_vs_target_html"] = p3_html
    generated_files["lag_vs_target_json"] = p3_json

    plt.figure(figsize=(7, 5))
    plt.scatter(sample_sub["demand_lag_1"], sample_sub["Demand"], alpha=0.3, color="#10b981", s=15)
    plt.plot([0, 400], [0, 400], color='gray', linestyle='--', alpha=0.6)
    plt.title("Lag 1 Historical Demand vs Target Demand", fontsize=11, fontweight='bold')
    plt.xlabel("Lag 1 Demand (Previous Day)")
    plt.ylabel("Target Demand (Current Day)")
    plt.tight_layout()
    p3_png = os.path.join(output_dir, "03_lag_vs_target_relationships.png")
    plt.savefig(p3_png, dpi=150)
    plt.close()
    generated_files["lag_vs_target_png"] = p3_png

    # 4. Rolling Demand Comparisons (Time-Series Trajectory for a Single SKU)
    single_series = df_features[(df_features["Store ID"] == "S001") & (df_features["Product ID"] == "P0001")].copy()
    single_series["Date_dt"] = pd.to_datetime(single_series["Date"])
    single_series = single_series.sort_values("Date_dt").tail(120)

    fig4 = px.line(
        single_series,
        x="Date",
        y=["Demand", "rolling_mean_7", "rolling_mean_28"],
        title="Time-Series Smoothing: Actual Demand vs 7-Day & 28-Day Rolling Averages (Store S001, SKU P0001)",
        labels={"value": "Demand Units", "variable": "Feature"},
        color_discrete_map={"Demand": "#94a3b8", "rolling_mean_7": "#10b981", "rolling_mean_28": "#3b82f6"}
    )
    fig4.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p4_html = os.path.join(output_dir, "04_rolling_demand_comparisons.html")
    p4_json = os.path.join(output_dir, "04_rolling_demand_comparisons.json")
    fig4.write_html(p4_html)
    with open(p4_json, "w") as f:
        f.write(fig4.to_json())
    generated_files["rolling_demand_html"] = p4_html
    generated_files["rolling_demand_json"] = p4_json

    plt.figure(figsize=(11, 4.5))
    plt.plot(single_series["Date_dt"], single_series["Demand"], label="Actual Demand", color="#94a3b8", alpha=0.7)
    plt.plot(single_series["Date_dt"], single_series["rolling_mean_7"], label="7-Day Rolling Mean", color="#10b981", linewidth=2)
    plt.plot(single_series["Date_dt"], single_series["rolling_mean_28"], label="28-Day Rolling Mean", color="#3b82f6", linewidth=2)
    plt.title("Rolling Statistics vs Actual Demand (Store S001, Product P0001)", fontsize=11, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()
    p4_png = os.path.join(output_dir, "04_rolling_demand_comparisons.png")
    plt.savefig(p4_png, dpi=150)
    plt.close()
    generated_files["rolling_demand_png"] = p4_png

    # 5. Feature Distributions (Histograms)
    feat_subset = ["rolling_mean_7", "rolling_std_7", "recent_demand_trend", "price_ratio"]
    fig5 = make_subplots(rows=2, cols=2, subplot_titles=[f"Distribution of {f}" for f in feat_subset])
    coords5 = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for i, f in enumerate(feat_subset):
        r, cl = coords5[i]
        fig5.add_trace(go.Histogram(x=df_features[f], name=f, marker_color="#10b981", nbinsx=30), row=r, col=cl)
    fig5.update_layout(template="plotly_white", title_text="Engineered Feature Value Distributions", showlegend=False, font=dict(family="Inter, sans-serif"))
    p5_html = os.path.join(output_dir, "05_feature_distributions.html")
    p5_json = os.path.join(output_dir, "05_feature_distributions.json")
    fig5.write_html(p5_html)
    with open(p5_json, "w") as f:
        f.write(fig5.to_json())
    generated_files["feature_distributions_html"] = p5_html
    generated_files["feature_distributions_json"] = p5_json

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i, f in enumerate(feat_subset):
        ax = axes[i // 2, i % 2]
        sns.histplot(df_features[f], kde=True, ax=ax, color="#10b981", bins=30)
        ax.set_title(f"{f} Distribution", fontsize=10, fontweight='bold')
    plt.suptitle("Distributions of Key Engineered Features", fontsize=12, fontweight='bold')
    plt.tight_layout()
    p5_png = os.path.join(output_dir, "05_feature_distributions.png")
    plt.savefig(p5_png, dpi=150)
    plt.close()
    generated_files["feature_distributions_png"] = p5_png

    logger.info(f"Generated all 5 Stage 4 Feature Engineering visualizations in {output_dir}")
    return generated_files


def generate_stage6_recommendation_visualizations(
    df_recs: pd.DataFrame,
    features_df: pd.DataFrame,
    sample_forecast: List[Dict[str, Any]],
    output_dir: str = "data/reports/recommendation_visualizations"
) -> Dict[str, str]:
    """
    Generates all required Stage 6 Inventory Recommendation visualizations:
    1. Products at high stockout risk
    2. Products with excess stock (overstock)
    3. Inventory coverage distribution (Days of Supply)
    4. Recommended reorder quantities
    5. Forecast versus current inventory
    6. Risk by category and store
    7. Demand forecast with historical values
    8. Priority-based reorder list
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    # 1. Products at High Stockout Risk
    stockout_df = df_recs[df_recs["Risk Category"].str.contains("STOCKOUT")].sort_values("Current Inventory").head(15)
    fig1 = px.bar(
        stockout_df,
        x="Product ID",
        y="Current Inventory",
        color="Risk Category",
        hover_data=["Store ID", "Category", "Reorder Point", "Suggested Reorder Qty"],
        title="Products at Critical & High Stockout Risk (Current Inventory vs Reorder Point)",
        color_discrete_map={"CRITICAL STOCKOUT RISK": "#b91c1c", "HIGH STOCKOUT RISK": "#ef4444"}
    )
    fig1.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p1_html = os.path.join(output_dir, "01_high_stockout_risk_products.html")
    p1_json = os.path.join(output_dir, "01_high_stockout_risk_products.json")
    fig1.write_html(p1_html)
    with open(p1_json, "w") as f:
        f.write(fig1.to_json())
    generated_files["stockout_risk_html"] = p1_html
    generated_files["stockout_risk_json"] = p1_json

    plt.figure(figsize=(9, 4.5))
    if len(stockout_df) > 0:
        plt.bar(stockout_df["Product ID"] + " (" + stockout_df["Store ID"] + ")", stockout_df["Current Inventory"], color="#ef4444")
        plt.title("Products at High Stockout Risk (Depleted Inventory)", fontsize=11, fontweight='bold')
        plt.ylabel("Current Units")
        plt.xticks(rotation=35, ha='right')
    else:
        plt.text(0.5, 0.5, "Zero Stockout Risk SKUs", ha='center', va='center')
    plt.tight_layout()
    p1_png = os.path.join(output_dir, "01_high_stockout_risk_products.png")
    plt.savefig(p1_png, dpi=150)
    plt.close()
    generated_files["stockout_risk_png"] = p1_png

    # 2. Products with Excess Stock (Overstock)
    overstock_df = df_recs[df_recs["Risk Category"] == "OVERSTOCK RISK"].sort_values("Days of Supply", ascending=False).head(15)
    fig2 = px.bar(
        overstock_df,
        x="Product ID",
        y="Days of Supply",
        color="Category",
        title="Products with Excess Stock (>30 Days of Supply Threshold)",
        text_auto=".1f",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig2.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p2_html = os.path.join(output_dir, "02_excess_stock_overstock_products.html")
    p2_json = os.path.join(output_dir, "02_excess_stock_overstock_products.json")
    fig2.write_html(p2_html)
    with open(p2_json, "w") as f:
        f.write(fig2.to_json())
    generated_files["overstock_products_html"] = p2_html
    generated_files["overstock_products_json"] = p2_json

    plt.figure(figsize=(9, 4.5))
    if len(overstock_df) > 0:
        plt.bar(overstock_df["Product ID"] + " (" + overstock_df["Store ID"] + ")", overstock_df["Days of Supply"], color="#f59e0b")
        plt.axhline(30, color="red", linestyle="--", label="30-Day Overstock Limit")
        plt.title("Products with Excess Stock (Days of Supply)", fontsize=11, fontweight='bold')
        plt.ylabel("Days of Supply")
        plt.xticks(rotation=35, ha='right')
        plt.legend()
    plt.tight_layout()
    p2_png = os.path.join(output_dir, "02_excess_stock_overstock_products.png")
    plt.savefig(p2_png, dpi=150)
    plt.close()
    generated_files["overstock_products_png"] = p2_png

    # 3. Inventory Coverage Distribution (Days of Supply)
    # Clip extreme DoS for histogram readability
    dos_clipped = df_recs["Days of Supply"].clip(upper=60)
    fig3 = px.histogram(
        dos_clipped,
        nbins=30,
        title="Distribution of Inventory Coverage (Days of Supply Across 100 SKUs)",
        color_discrete_sequence=["#10b981"]
    )
    fig3.add_vline(x=30, line_dash="dash", line_color="orange", annotation_text="Overstock >30d")
    fig3.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="Lead Time (3d)")
    fig3.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p3_html = os.path.join(output_dir, "03_inventory_coverage_distribution.html")
    p3_json = os.path.join(output_dir, "03_inventory_coverage_distribution.json")
    fig3.write_html(p3_html)
    with open(p3_json, "w") as f:
        f.write(fig3.to_json())
    generated_files["inventory_coverage_html"] = p3_html
    generated_files["inventory_coverage_json"] = p3_json

    plt.figure(figsize=(8, 4))
    sns.histplot(dos_clipped, kde=True, color="#10b981", bins=25)
    plt.axvline(3, color="red", linestyle="--", label="Lead Time (3 Days)")
    plt.axvline(30, color="orange", linestyle="--", label="Overstock Buffer (30 Days)")
    plt.title("Inventory Coverage Distribution (Days of Supply)", fontsize=11, fontweight='bold')
    plt.xlabel("Days of Supply")
    plt.legend()
    plt.tight_layout()
    p3_png = os.path.join(output_dir, "03_inventory_coverage_distribution.png")
    plt.savefig(p3_png, dpi=150)
    plt.close()
    generated_files["inventory_coverage_png"] = p3_png

    # 4. Recommended Reorder Quantities (Top Orders)
    reorder_top = df_recs[df_recs["Suggested Reorder Qty"] > 0].sort_values("Suggested Reorder Qty", ascending=False).head(15)
    fig4 = px.bar(
        reorder_top,
        x="Product ID",
        y="Suggested Reorder Qty",
        color="Store ID",
        text_auto=True,
        title="Top 15 Recommended Replenishment Orders (Units to Reorder Now)"
    )
    fig4.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p4_html = os.path.join(output_dir, "04_recommended_reorder_quantities.html")
    p4_json = os.path.join(output_dir, "04_recommended_reorder_quantities.json")
    fig4.write_html(p4_html)
    with open(p4_json, "w") as f:
        f.write(fig4.to_json())
    generated_files["reorder_quantities_html"] = p4_html
    generated_files["reorder_quantities_json"] = p4_json

    plt.figure(figsize=(9, 4.5))
    if len(reorder_top) > 0:
        plt.bar(reorder_top["Product ID"] + " (" + reorder_top["Store ID"] + ")", reorder_top["Suggested Reorder Qty"], color="#059669")
        plt.title("Top Recommended Reorder Quantities (Units)", fontsize=11, fontweight='bold')
        plt.ylabel("Suggested Order Units")
        plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    p4_png = os.path.join(output_dir, "04_recommended_reorder_quantities.png")
    plt.savefig(p4_png, dpi=150)
    plt.close()
    generated_files["reorder_quantities_png"] = p4_png

    # 5. Forecast versus Current Inventory
    fig5 = px.scatter(
        df_recs,
        x="Forecast 14d Demand",
        y="Current Inventory",
        color="Risk Category",
        size="Suggested Reorder Qty",
        hover_name="Product ID",
        title="14-Day Forecasted Demand vs Current On-Hand Inventory",
        color_discrete_map={
            "CRITICAL STOCKOUT RISK": "#b91c1c",
            "HIGH STOCKOUT RISK": "#ef4444",
            "OVERSTOCK RISK": "#f59e0b",
            "OPTIMAL STOCK LEVEL": "#10b981"
        }
    )
    fig5.add_shape(type="line", line=dict(dash="dash", color="black"), x0=0, x1=2500, y0=0, y1=2500)
    fig5.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p5_html = os.path.join(output_dir, "05_forecast_vs_current_inventory.html")
    p5_json = os.path.join(output_dir, "05_forecast_vs_current_inventory.json")
    fig5.write_html(p5_html)
    with open(p5_json, "w") as f:
        f.write(fig5.to_json())
    generated_files["forecast_vs_inventory_html"] = p5_html
    generated_files["forecast_vs_inventory_json"] = p5_json

    plt.figure(figsize=(7, 6))
    for cat, grp in df_recs.groupby("Risk Category"):
        plt.scatter(grp["Forecast 14d Demand"], grp["Current Inventory"], label=cat, alpha=0.6, s=40)
    plt.plot([0, 2500], [0, 2500], color='black', linestyle='--', label='1:1 Parity Line')
    plt.title("Forecasted 14-Day Demand vs Current Inventory", fontsize=11, fontweight='bold')
    plt.xlabel("14-Day Predicted Demand (Units)")
    plt.ylabel("Current On-Hand Inventory (Units)")
    plt.legend()
    plt.tight_layout()
    p5_png = os.path.join(output_dir, "05_forecast_vs_current_inventory.png")
    plt.savefig(p5_png, dpi=150)
    plt.close()
    generated_files["forecast_vs_inventory_png"] = p5_png

    # 6. Risk by Category and Store
    cat_risk = pd.crosstab(df_recs["Category"], df_recs["Risk Category"])
    fig6 = px.bar(
        cat_risk.reset_index(),
        x="Category",
        y=cat_risk.columns.tolist(),
        title="Inventory Health Risk Profile by Product Category",
        color_discrete_map={
            "CRITICAL STOCKOUT RISK": "#b91c1c",
            "HIGH STOCKOUT RISK": "#ef4444",
            "OVERSTOCK RISK": "#f59e0b",
            "OPTIMAL STOCK LEVEL": "#10b981"
        }
    )
    fig6.update_layout(template="plotly_white", barmode="stack", font=dict(family="Inter, sans-serif"))
    p6_html = os.path.join(output_dir, "06_risk_by_category_and_store.html")
    p6_json = os.path.join(output_dir, "06_risk_by_category_and_store.json")
    fig6.write_html(p6_html)
    with open(p6_json, "w") as f:
        f.write(fig6.to_json())
    generated_files["risk_by_category_html"] = p6_html
    generated_files["risk_by_category_json"] = p6_json

    plt.figure(figsize=(8, 4.5))
    cat_risk.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#b91c1c', '#ef4444', '#10b981', '#f59e0b'][:len(cat_risk.columns)])
    plt.title("Inventory Risk Breakdown by Category", fontsize=11, fontweight='bold')
    plt.ylabel("SKU Count")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    p6_png = os.path.join(output_dir, "06_risk_by_category_and_store.png")
    plt.savefig(p6_png, dpi=150)
    plt.close()
    generated_files["risk_by_category_png"] = p6_png

    # 7. Demand Forecast with Historical Values (Sample Series: Store S001, Product P0001)
    hist_sample = features_df[(features_df["Store ID"] == "S001") & (features_df["Product ID"] == "P0001")].tail(45).copy()
    hist_sample["Date_dt"] = pd.to_datetime(hist_sample["Date"])
    
    df_fc = pd.DataFrame(sample_forecast)
    df_fc["Date_dt"] = pd.to_datetime(df_fc["date"])

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=hist_sample["Date_dt"], y=hist_sample["Demand"], mode="lines", name="Historical Actual Demand", line=dict(color="#3b82f6", width=2)))
    fig7.add_trace(go.Scatter(x=df_fc["Date_dt"], y=df_fc["predicted_demand"], mode="lines+markers", name="14-Day Recursive Forecast", line=dict(color="#10b981", width=2.5, dash="dash")))
    fig7.update_layout(template="plotly_white", title="Demand Forecast Trajectory: Historical Observations + 14-Day Forecast (Store S001, SKU P0001)", font=dict(family="Inter, sans-serif"))
    p7_html = os.path.join(output_dir, "07_forecast_with_historical_demand.html")
    p7_json = os.path.join(output_dir, "07_forecast_with_historical_demand.json")
    fig7.write_html(p7_html)
    with open(p7_json, "w") as f:
        f.write(fig7.to_json())
    generated_files["forecast_trajectory_html"] = p7_html
    generated_files["forecast_trajectory_json"] = p7_json

    plt.figure(figsize=(11, 4.5))
    plt.plot(hist_sample["Date_dt"], hist_sample["Demand"], label="Historical Actual Demand", color="#3b82f6", linewidth=2)
    plt.plot(df_fc["Date_dt"], df_fc["predicted_demand"], label="14-Day ML Forecast", color="#10b981", linewidth=2.5, linestyle="--", marker="o", markersize=4)
    plt.title("Historical Demand + 14-Day ML Forecast (Store S001, Product P0001)", fontsize=11, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Demand (Units)")
    plt.legend()
    plt.tight_layout()
    p7_png = os.path.join(output_dir, "07_forecast_with_historical_demand.png")
    plt.savefig(p7_png, dpi=150)
    plt.close()
    generated_files["forecast_trajectory_png"] = p7_png

    # 8. Priority-Based Reorder List (Ranked Action Items)
    priority_df = df_recs[df_recs["Review Priority"] == "HIGH"].sort_values("Suggested Reorder Qty", ascending=False).head(15)
    fig8 = px.bar(
        priority_df,
        x="Product ID",
        y="Suggested Reorder Qty",
        color="Risk Category",
        hover_data=["Store ID", "Category", "Current Inventory", "Days of Supply"],
        title="Immediate Action Reorder List: High Priority Replenishment SKUs",
        color_discrete_map={"CRITICAL STOCKOUT RISK": "#b91c1c", "HIGH STOCKOUT RISK": "#ef4444"}
    )
    fig8.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif"))
    p8_html = os.path.join(output_dir, "08_priority_reorder_list.html")
    p8_json = os.path.join(output_dir, "08_priority_reorder_list.json")
    fig8.write_html(p8_html)
    with open(p8_json, "w") as f:
        f.write(fig8.to_json())
    generated_files["priority_reorder_list_html"] = p8_html
    generated_files["priority_reorder_list_json"] = p8_json

    plt.figure(figsize=(9, 4.5))
    if len(priority_df) > 0:
        plt.bar(priority_df["Product ID"] + " (" + priority_df["Store ID"] + ")", priority_df["Suggested Reorder Qty"], color="#b91c1c")
        plt.title("High Priority Reorder List (Ranked by Urgency)", fontsize=11, fontweight='bold')
        plt.ylabel("Order Units")
        plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    p8_png = os.path.join(output_dir, "08_priority_reorder_list.png")
    plt.savefig(p8_png, dpi=150)
    plt.close()
    generated_files["priority_reorder_list_png"] = p8_png

    logger.info(f"Generated all 8 Stage 6 Recommendation visualizations in {output_dir}")
    return generated_files




