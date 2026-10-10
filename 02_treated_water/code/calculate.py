from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def calculate():
    source = pd.read_csv(ROOT / 'results/Figure3_source_data.csv')
    source['cat_component_digitized'] = pd.to_numeric(source.cat_component_digitized)
    source['enrichment_ratio'] = source.cat_pct_digitized / source.concentration_pct.replace(0, float('nan'))
    source['cat_fraction'] = source.cat_pct_digitized / 100
    source['concentration_fraction'] = source.concentration_pct / 100
    return source


if __name__ == '__main__':
    out = calculate()
    out.to_csv(ROOT / 'results/class_contributions.csv', index=False)
    print(out.groupby('category')[['concentration_fraction', 'cat_fraction']].mean().to_string())
