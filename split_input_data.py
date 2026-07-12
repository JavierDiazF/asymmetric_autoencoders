#! /usr/bin/python3

import pandas as pd

CSV_PATH = 'Datasets/Caples_Lake_N7_2014_2017.csv'
TRAIN_PATH = 'Datasets/Caples_Lake_N7_2014_2017_train.csv'
VAL_PATH = 'Datasets/Caples_Lake_N7_2014_2017_val.csv'
TEST_PATH = 'Datasets/Caples_Lake_N7_2014_2017_test.csv'

def count_per_year(csv_path):
    # Solo cargamos la columna de fecha (no la de temperatura) para que sea rapido
    raw = pd.read_csv(csv_path, usecols=[0]).iloc[:, 0]
    dates = pd.to_datetime(raw, format='mixed', errors='coerce')

    bad = dates.isna()
    if bad.any():
        print(f'Warning: {bad.sum()} rows not parseabled (ignored):')
        print(raw[bad].head().to_string(index=True))
        print()
        dates = dates[~bad]

    counts = dates.dt.year.value_counts().sort_index()
    return counts, dates


def chronological_split(dates, val_frac=0.08, test_year=2017):
    """Divide 'dates' en train/val/test respetando el orden cronologico.

    - test: todo lo que sea >= test_year (aqui, 2017 entero).
    - val: el ultimo val_frac (sobre el total) de lo que quede antes de test_year.
    - train: el resto (todo lo anterior al recorte de val).

    Devuelve, para cada split, el rango de filas [row_start, row_end] (ambos
    incluidos, numeracion 1-based tal que 1 == primera fila de datos del CSV,
    igual convencion que 'start' en get_window_starts/read_window).
    """
    n_total = len(dates)
    n_val = round(val_frac * n_total)

    is_test = dates.dt.year >= test_year
    trainval_idx = dates.index[~is_test]
    test_idx = dates.index[is_test]

    val_idx = trainval_idx[-n_val:]
    train_idx = trainval_idx[:-n_val]

    splits = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        row_start = idx.min() + 1  # +1: index 0 del DataFrame == fila 1 del CSV (tras la cabecera)
        row_end = idx.max() + 1
        splits[name] = (row_start, row_end)
    return splits

def save_splits(csv_path, train_path, val_path, test_path, splits, out_dir='.'):
    df = pd.read_csv(csv_path)  # header=0 por defecto -> columnas ya con sus nombres
    destination_dict = {'train': train_path, 'val': val_path, 'test': test_path}
    for name in ('train', 'val', 'test'):
        row_start, row_end = splits[name]
        chunk = df.iloc[row_start - 1: row_end]  # -1: row_start es 1-based, iloc es 0-based
        chunk.to_csv(destination_dict[name], index=False)

def split_input_csv(original_path, train_path, val_path, test_path):
    counts, dates = count_per_year(original_path)

    print(f'Date range: {dates.min()}  -->  {dates.max()}')
    print(f'Total measures: {len(dates)}')
    print()
    print('Measures per year:')
    for year, n in counts.items():
        pct = 100 * n / len(dates)
        print(f'  {year}: {n:>7} measures  ({pct:5.1f}% of total)')

    print()
    print('Chronologic split (train 70% / val 8% / test = year 2017):')
    splits = chronological_split(dates, val_frac=0.08, test_year=2017)
    n_total = len(dates)
    for name in ('train', 'val', 'test'):
        row_start, row_end = splits[name]
        n_rows = row_end - row_start + 1
        pct = 100 * n_rows / n_total
        start_date = dates.loc[row_start - 1]
        end_date = dates.loc[row_end - 1]
        print(f'  {name:>5}: filas {row_start:>6}-{row_end:<6} ({n_rows:>7} measures, {pct:5.1f}%) [{start_date} -> {end_date}]')
    save_splits(original_path, train_path, val_path, test_path, splits, out_dir='Datasets')

if __name__ == '__main__':
    split_input_csv(CSV_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH)
    