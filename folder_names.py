import pandas as pd


def filter_csv_by_ids(input_csv, output_csv, ids_list, id_column_name):
    """
    Filters rows in the input CSV file based on a list of IDs and saves the filtered rows to a new CSV file.

    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to the output CSV file where filtered rows will be saved.
    :param ids_list: List of IDs to filter by.
    :param id_column_name: The name of the column to filter by.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_csv)

        # Filter rows where the specified ID column is in the ids_list
        if id_column_name in df.columns:
            filtered_df = df[df[id_column_name].isin(ids_list)]

            # Save the filtered DataFrame to a new CSV file
            filtered_df.to_csv(output_csv, index=False)
            print(f"Filtered CSV saved to {output_csv}")
        else:
            print(f"Column '{id_column_name}' not found in the input CSV.")
    except FileNotFoundError:
        print(f"File '{input_csv}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
input_csv = 'D:\\Voice Authetication ESPNET\\datasetvox\\vox1_meta.csv'
output_csv = 'D:\\new.csv'
ids_list = ['id10002', 'id10003', 'id10017', 'id10018', 'id10045', 'id10324', 'id10393', 'id10519', 'id10583', 'id10724', 'id10852', 'id10901',
            'id10912', 'id10941', 'id10943', 'id10955', 'id10956', 'id11071', 'id11089', 'id11090', 'id11100', 'id11130', 'id11136', 'id11209']
# Replace with the correct column name found from check_csv_columns
id_column_name = 'VID'

# Call the function to filter the CSV
filter_csv_by_ids(input_csv, output_csv, ids_list, id_column_name)
