from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def load_split_csvs(directory):
    all_files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


dataset_path_old = "/Users/andre/Desktop/INAF/USA/dataset/code/SOM/SOM-CSC/data/csc211_mastertable_clean_observationlevel_COMPLETE_xmatchSimbad1arcsec_log_norm_id_ClassAgg/oldlabel"
dataset_path_new = "/Users/andre/Desktop/INAF/USA/dataset/code/SOM/SOM-CSC/data/csc211_mastertable_clean_observationlevel_COMPLETE_xmatchSimbad1arcsec_log_norm_id_ClassAgg/newlabel"

df_old = load_split_csvs(dataset_path_old)
df_new = load_split_csvs(dataset_path_new)

print(df_old['class'].unique())
print(df_new['class'].unique())

# print the difference classes name
print(set(df_old['class']) - set(df_new['class']))
print(set(df_new['class']) - set(df_old['class']))

exit()

# change YSO in neoYSOs and AGN in neoAGNs
df['class'] = df['class'].replace('YSO', 'neoYSOs')
df['class'] = df['class'].replace('AGN', 'neoAGNs')

print(df['class'].unique())

# Split the dataframe back into the same number of files as originally
num_files = len(glob.glob(os.path.join(dataset_path, '*.csv')))
rows_per_file = len(df) // num_files + 1

for i in range(num_files):
    start_idx = i * rows_per_file
    end_idx = min((i + 1) * rows_per_file, len(df))

    if start_idx < len(df):
        chunk_df = df.iloc[start_idx:end_idx]
        output_filename = os.path.join(dataset_path, f'output2_{i+1}.csv')
        chunk_df.to_csv(output_filename, index=False)
        print(f"Saved {output_filename} with {len(chunk_df)} rows")

print(f"\nCompleted! Added 'class' column to all {num_files} CSV files.")
print(f"Total rows processed: {len(df)}")

exit()

# read main_type column
main_type = df['main_type']
otype = df['otype']

print(main_type.unique())
print(otype.unique())

print("YSO")
index_NewYSO = main_type[main_type == 'YSO'] + main_type[main_type ==
                                                         'OrionV*'] + main_type[main_type == 'TTauri*']
print(len(index_NewYSO))

print("Binaries")

index_Binaries = main_type[main_type == 'HighMassXBin'] + main_type[main_type == 'XrayBin'] + main_type[main_type ==
                                                                                                        'LowMassXBin'] + main_type[main_type == 'CataclyV*'] + main_type[main_type == 'SB*'] + main_type[main_type == 'BYDraV*']
print(len(index_Binaries))

print("Stars")
index_Stars = main_type[main_type == 'Star']
print(len(index_Stars))


print("AGN")

index_AGN = main_type[main_type == 'AGN'] + main_type[main_type == 'QSO'] + main_type[main_type ==
                                                                                      'Seyfert1'] + main_type[main_type == 'Seyfert2'] + main_type[main_type == 'BLLac']
print(len(index_AGN))

print("X")
index_X = main_type[main_type == 'X'] + main_type[main_type == 'ULX']
print(len(index_X))

print("Galaxies")

index_Galaxies = main_type[main_type ==
                           'PartofG'] + main_type[main_type == 'Galaxy']
print(len(index_Galaxies))


print("GC")
index_GC = main_type[main_type == 'GlobCluster'] + \
    main_type[main_type == 'Pulsar']

print(len(index_GC))

# Create the class column mapping


def create_class_column(df):
    """
    Create a new 'class' column based on main_type aggregations
    """
    # Initialize the class column with the original main_type
    df['class'] = df['main_type'].copy()

    # Define the mapping dictionary
    class_mapping = {
        # YSO class
        'YSO': 'YSO',
        'OrionV*': 'YSO',
        'TTauri*': 'YSO',

        # Binaries class
        'HighMassXBin': 'Binaries',
        'XrayBin': 'Binaries',
        'LowMassXBin': 'Binaries',
        'CataclyV*': 'Binaries',
        'SB*': 'Binaries',
        'BYDraV*': 'Binaries',

        # Stars class (unchanged)
        'Star': 'Stars',

        # AGN class
        'AGN': 'AGN',
        'QSO': 'AGN',
        'Seyfert1': 'AGN',
        'Seyfert2': 'AGN',
        'BLLac': 'AGN',

        # X class
        'X': 'X',
        'ULX': 'X',

        # Galaxies class
        'PartofG': 'Galaxies',
        'Galaxy': 'Galaxies',

        # GC class
        'GlobCluster': 'GC',
        'Pulsar': 'GC'
    }

    # Apply the mapping
    df['class'] = df['main_type'].map(class_mapping).fillna(df['main_type'])

    return df


# Apply the class column creation
print("\nCreating class column...")
df = create_class_column(df)

# Display the mapping results
print("\nClass distribution:")
print(df['class'].value_counts())

print("\nOriginal main_type vs new class mapping:")
mapping_df = df.groupby(['main_type', 'class']
                        ).size().reset_index(name='count')
print(mapping_df)

# Save the updated dataset back to CSV files
print("\nSaving updated dataset with 'class' column...")


# Split the dataframe back into the same number of files as originally
num_files = len(glob.glob(os.path.join(dataset_path, '*.csv')))
rows_per_file = len(df) // num_files + 1

for i in range(num_files):
    start_idx = i * rows_per_file
    end_idx = min((i + 1) * rows_per_file, len(df))

    if start_idx < len(df):
        chunk_df = df.iloc[start_idx:end_idx]
        output_filename = os.path.join(dataset_path, f'output_{i+1}.csv')
        chunk_df.to_csv(output_filename, index=False)
        print(f"Saved {output_filename} with {len(chunk_df)} rows")

print(f"\nCompleted! Added 'class' column to all {num_files} CSV files.")
print(f"Total rows processed: {len(df)}")

# Display sample of the new data structure
print("\nSample of data with new 'class' column:")
sample_cols = ['id', 'name', 'main_type',
               'class'] if 'name' in df.columns else ['id', 'main_type', 'class']
if all(col in df.columns for col in sample_cols):
    print(df[sample_cols].head(10))
