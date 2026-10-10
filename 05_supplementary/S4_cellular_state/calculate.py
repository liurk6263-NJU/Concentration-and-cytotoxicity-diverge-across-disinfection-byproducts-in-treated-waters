from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def matrices():
    cross_cell = pd.read_csv(ROOT/'cross_cell.csv')
    pathway = pd.read_csv(ROOT/'pathways.csv')
    return cross_cell,pathway


if __name__ == '__main__':
    cross_cell,pathway = matrices()
    print({'cross_cell_rows':len(cross_cell),'pathway_rows':len(pathway)})
