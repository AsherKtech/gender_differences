'''
ESS PIPELINE: DUAL INDICATOR FLAGS (HIST-GRADIENT BOOSTING & NATIVE SPLITS)
====================================================================================================
'''


#%%
import os
import pandas as pd
import plotly.graph_objects as go
from pyprojroot import here
#%%


def create_unified_countries_plot6(perf_df, output_html="countries_gender_similarity_trends_unified6.html", min_rounds=1, country_map=None):
    """
    Generates a unified Plotly line plot for a subset of countries passed via country_map dictionary.
    Dynamically extracts custom keywords from output_html filename to append to the title,
    preserving underscores and specific suffix tags.
    """
    round_counts = perf_df.groupby('Country')['ESS_round'].nunique()
    eligible_countries = round_counts[round_counts >= min_rounds].index
    
    # Filter by country_map keys if provided, otherwise use all eligible countries
    if country_map is not None:
        target_countries = [c for c in country_map.keys() if c in eligible_countries]
    else:
        target_countries = list(eligible_countries)
        
    filtered_df = perf_df[perf_df['Country'].isin(target_countries)].copy()

    # --- Dynamic Title Parsing (Preserving Underscores) ---
    filename_base = os.path.splitext(os.path.basename(output_html))[0]
    
    # Standard base prefixes to strip out completely
    base_prefix = "countries_gender_similarity_trends_"
    
    # Words/tokens to ignore if present
    ignored_words = {
        "countries", "gender", "similarity", 
        "trends", "predictability", "accuracy", "plot", "unified6"
    }
    
    if filename_base.startswith(base_prefix):
        suffix_raw = filename_base[len(base_prefix):]
    else:
        # Fallback: split by '_' and keep parts not in ignored_words
        parts = filename_base.split('_')
        suffix_raw = "_".join([p for p in parts if p.lower() not in ignored_words])

    if suffix_raw:
        suffix_label = f" {suffix_raw}"
    else:
        suffix_label = ""
        
    title_text = f"<b>Gender Predictability Trends{suffix_label}</b><br><sup>Lower Accuracy = Higher Gender Similarity</sup>"

    print(f"\nGenerating unified plot for {len(target_countries)} countries -> '{output_html}'")

    # Order countries by descending mean accuracy for legend layout
    country_means = filtered_df.groupby('Country')['accuracy'].mean().sort_values(ascending=False)
    country_order = country_means.index

    fig = go.Figure()

    # 1. Individual country traces
    for cntry in country_order:
        cntry_df = filtered_df[filtered_df['Country'] == cntry]
        cntry_sorted = cntry_df.sort_values('ESS_round')
        cntry_mean = country_means[cntry]
        
        trace_kwargs = dict(
            x=cntry_sorted['ESS_round'],
            y=cntry_sorted['accuracy'],
            mode='lines+markers',
            name=cntry,
            opacity=0.85,
            hovertemplate=(
                f"<b>Country:</b> {cntry}<br>"
                f"<b>Round:</b> %{{x}}<br>"
                f"<b>Accuracy:</b> %{{y:.2%}}<br>"
                f"<b>Mean Acc:</b> {cntry_mean:.2%}"
                "<extra></extra>"
            )
        )
        
        # Apply color mapping
        if country_map and cntry in country_map and country_map[cntry]:
            trace_kwargs['line'] = dict(color=country_map[cntry], width=2)
            trace_kwargs['marker'] = dict(color=country_map[cntry], size=6)

        fig.add_trace(go.Scatter(**trace_kwargs))

    # 2. Average accuracy trend line across subset countries per round
    avg_accuracy_by_round = (
        filtered_df.groupby('ESS_round')['accuracy']
        .mean()
        .reset_index()
        .sort_values('ESS_round')
    )

    fig.add_trace(go.Scatter(
        x=avg_accuracy_by_round['ESS_round'],
        y=avg_accuracy_by_round['accuracy'],
        mode='lines+markers',
        name='<b>Group Average</b>',
        line=dict(color='black', width=3.5),
        marker=dict(size=7, color='black'),
        hovertemplate="<b>GROUP AVERAGE</b><br><b>Round:</b> %{x}<br><b>Accuracy:</b> %{y:.2%}<extra></extra>"
    ))

    # 3. Overall subset mean accuracy
    overall_avg_accuracy = filtered_df['accuracy'].mean()

    # 4. Layout configuration
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        title=title_text,
        xaxis=dict(
            title="ESS Survey Round",
            dtick=1,
            tickmode='linear',
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            ticks="outside",
            tickcolor="black",
            ticklen=6,
            tickwidth=1.5
        ),
        yaxis=dict(
            title="Model Accuracy",
            tickformat=".0%",
            range=[0.65, 0.90],
            showgrid=False,
            zeroline=False
        ),
        height=700,
        width=1200,
        hovermode="closest",
        margin=dict(l=80, r=120, t=100, b=80)
    )

    # Reference line for group grand mean
    fig.add_hline(
        y=overall_avg_accuracy, 
        line_dash="dot", 
        line_color="black", 
        line_width=2,
        annotation_text=f"Group Mean ({overall_avg_accuracy:.2%})", 
        annotation_position="top right",
        annotation_font_color="black"
    )

    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"✅ Saved plot to '{output_html}' (Overall Mean Accuracy: {overall_avg_accuracy:.4f})")



    
#%%
# Save performance metrics to CSV for downstream analysis
perf_df = pd.read_csv(here("data/processed/gender_country_and_round_performance_indicators.csv"))
perf_df


#%%
# ==================================================================================================
# UNIFIED COLOR PALETTE
# Shared 10-color high-contrast palette across both country groups
# ==================================================================================================

# Group 1: North & Western Europe
nw_europe_map = {
    'Netherlands':    '#E6194B',  # Red
    'Switzerland':    "#209B31",  # Green
    'Germany':        '#FFE119',  # Yellow
    'Ireland':        '#4363D8',  # Blue
    'Norway':         '#F58231',  # Orange
    'Belgium':        '#911EB4',  # Purple
    'United Kingdom': '#42D4F4',  # Cyan
    'Denmark':        '#F032E6',  # Magenta
    'France':         "#00FD0D",  # Teal
    'Sweden':         '#9A6324'   # Brown
}

# Group 2: Southern, Central & Eastern Europe + Israel
other_countries_map = {
    'Spain':    '#E6194B',  # Red
    'Slovenia':  "#209B31",  # Green
    'Austria':  '#FFE119',  # Yellow
    'Portugal': '#4363D8',  # Blue
    'Poland':   '#F58231',  # Orange
    'Israel':   '#911EB4',  # Purple
    'Estonia':  '#42D4F4',  # Cyan
    'Hungary':  '#F032E6',  # Magenta
    'Slovakia': "#00FD0D",  # Teal
    'Czechia':  '#9A6324'   # Brown
}

#%%
# 1. Run for North and Western Europe
create_unified_countries_plot6(
    perf_df, 
    output_html=here("plots/countries_gender_similarity_trends_nw_europe.html"), 
    country_map=nw_europe_map
)

# 2. Run for Remaining Countries
create_unified_countries_plot6(
    perf_df, 
    output_html=here("plots/countries_gender_similarity_trends_other.html"),  
    country_map=other_countries_map
)
# %%
