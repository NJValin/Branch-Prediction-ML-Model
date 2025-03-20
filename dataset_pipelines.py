import csv
import pandas as pd
from rich.progress import Progress
from icecream import ic

def combine_columns(row : dict)->dict:
        processed_row = {}
        processed_row['PC'] = int('0b'+''.join([str(int(float(row[str(row_index)]))) for row_index in range(0,32)]), 2)
        for index in range(32, 96):
            processed_row[f'GSHARE_TABLE_{index-32}'] = row[str(index)]

        base_col = 96
        for k in range(1, 49):
            processed_row[f'GA_TABLE_{k-1}'] = float(int('0b'+''.join([str(int(float(row[str(row_index)]))) for row_index in range(base_col+8*(k-1), base_col+(8*k))]), 2))

        processed_row["taken"] = int(float(row["480"]))

        return processed_row

def process_csv(input_csv_path : str, output_csv_path : str):
    i = 0
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile, \
         Progress() as progress:

        task = progress.add_task("[cyan]Processing...", total=100)

        reader = csv.DictReader(infile)
        first_row = combine_columns(next(reader))


        writer = csv.DictWriter(outfile, fieldnames=first_row.keys())
        writer.writeheader()

        writer.writerow(first_row)
        for row in reader:
            row_to_write = combine_columns(row)
            writer.writerow(row_to_write)
            if i%100_000 ==0:
                progress.advance(task, advance=1)

def balance_dataset(df, minority_class, majority_class, label_column):
    df_only_majority = df[df[label_column]==majority_class]
    df_only_minority = df[df[label_column]==minority_class]
    num_of_minority = (df[label_column].values == minority_class).sum()
    sampled_df = df_only_majority.sample(n=num_of_minority)
    balanced_df = pd.concat([sampled_df, df_only_minority]).reset_index(drop=True)
    return balanced_df.sample(frac=1).reset_index(drop=True)


