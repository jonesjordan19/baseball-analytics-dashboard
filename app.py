def render_100pct_stacked_bar(df_data, category_col, fixed_order, show_legend=False):
    """Draws an exact 100% horizontal stacked bar chart without title/legend collision."""
    if df_data.empty or category_col not in df_data.columns:
        return go.Figure()

    ctab = pd.crosstab(df_data[category_col], df_data['PitchFamily'], normalize='index').multiply(100)
    for col in ['Fastballs', 'Breaking', 'Offspeed', 'UD']:
        if col not in ctab.columns:
            ctab[col] = 0.0

    valid_order = [cat for cat in fixed_order if cat in ctab.index]
    ctab = ctab.reindex(valid_order).fillna(0.0)

    fig = go.Figure()
    for family in ['Fastballs', 'Breaking', 'Offspeed', 'UD']:
        fig.add_trace(go.Bar(
            y=ctab.index,
            x=ctab[family],
            name=family,
            orientation='h',
            marker=dict(color=PITCH_FAMILY_COLORS[family]),
            showlegend=show_legend,
            hoverinfo='text',
            hovertext=[f"{y} | {family}: {val:.1f}%" for y, val in zip(ctab.index, ctab[family])]
        ))

    fig.update_layout(
        barmode='stack',
        height=310,
        margin=dict(l=10, r=15, t=15, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(size=11)
        ),
        xaxis=dict(
            range=[0, 100],
            ticksuffix="%",
            dtick=25,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color="#111827", family="Arial Black")
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )
    return fig
