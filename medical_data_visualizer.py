import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['overweight'] > 25).astype(int)


# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    # 6
    df_cat = df_cat.value_counts(['cardio', 'variable', 'value']).reset_index(name='total')

    # 7
    cat_order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        order=cat_order
    )

    # 8
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig

# 10: Draw the Heat Map in the draw_heat_map function
def draw_heat_map():
    # 11: Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12: Calculate correlation matrix
    corr = df_heat.corr()

    # 13: Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15: Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        vmax=0.32,
        vmin=-0.16,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    # 16: Save figure
    fig.savefig('heatmap.png')
    return fig
