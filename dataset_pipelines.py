import csv

def combine_columns(row : dict)->dict:
        processed_row = {}
        processed_row['PC'] = int('0b'+''.join([str(int(float(row[str(row_index)]))) for row_index in range(0,32)]), 2)
        for index in range(32, 96):
            processed_row[f'GSHARE_TABLE_{index-32}'] = row[str(index)]

        base_col = 96
        for k in range(1, 49):
            processed_row[f'GA_TABLE_{k-1}'] = int('0b'+''.join([str(int(float(row[str(row_index)]))) for row_index in range(base_col+8*(k-1), base_col+(8*k))]), 2)

        processed_row["taken"] = int(float(row["480"]))

        return processed_row

def process_csv(input_csv_path : str, output_csv_path : str):
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        first_row = combine_columns(next(reader))


        writer = csv.DictWriter(outfile, fieldnames=first_row.keys())
        writer.writeheader()

        writer.writerow(first_row)
        for row in reader:
            row_to_write = combine_columns(row)
            writer.writerow(row_to_write)



if __name__=='__main__':
    process_csv("./csv/dataset_B/I04.csv", "./csv/processed_B/processed_I04.csv")
    process_csv("./csv/dataset_B/S02.csv", "./csv/processed_B/processed_S02.csv")
    process_csv("./csv/dataset_B/S04.csv", "./csv/processed_B/processed_S04.csv")
    process_csv("./csv/dataset_B/MM03.csv", "./csv/processed_B/processed_MM03.csv")
    process_csv("./csv/dataset_B/MM05.csv", "./csv/processed_B/processed_MM05.csv")
    process_csv("./csv/dataset_B/INT03.csv", "./csv/processed_B/processed_INT03.csv")
