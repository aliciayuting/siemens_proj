import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import numpy as np

warnings.filterwarnings("ignore")

suffix = ".dat"

def get_log_files(local_dir, suffix):
     log_files = []
     for root, dirs, files in os.walk(local_dir):
          for file in files:
               if file[-4:] == suffix:
                    file_path = os.path.join(root, file)
                    log_files.append(file_path)
     return log_files


def get_log_files_data(log_files):
     log_data = []
     for log_file_path in log_files:
          with open(log_file_path, 'r') as file:
               for line in file:
                    line = line.strip()
                    elements = line.split(' ') # remove the last element, which is extra info
                    log_data.append(elements)
     return log_data


def process_log_data_to_dataframe(log_data, drop_warmup=False, drop_model_info=False, filter_unifinished_job_instances=True):
     df = pd.DataFrame(log_data, columns=["tag", "node_id", "object_id", "timestamp", "extra"])
     df['tag'] = df['tag'].astype(int)
     df['node_id'] = df['node_id'].astype(int)
     df['object_id'] = df['object_id'].astype(int)
     df['extra'] = df['extra'].astype(int)
     df['roundID'] = df['extra'] // 1000
     df['cameraID'] = df['extra'] % 1000
     if drop_warmup:
          df_filtered = df[df['object_id'] >= 19]
          df = df_filtered
     df['timestamp'] = df['timestamp'].astype(int)
     df['timestamp'] = df['timestamp'] / 1000000 # convert to milliseconds
     min_value = df['timestamp'].min()
     df['timestamp'] -= min_value
     df['start_time'] = df.groupby(['object_id'])['timestamp'].transform(min)
     df['end_time'] = df.groupby(['object_id'])['timestamp'].transform(max)
     df['end_to_end_latency(ms)'] = df['end_time'] - df['start_time']
     return df


local_dir = "./collected_data"
log_files = get_log_files(local_dir, suffix)
log_data = get_log_files_data(log_files)

df = process_log_data_to_dataframe(log_data)


# Set options to display all rows and columns
pd.set_option('display.max_rows', None)  # Replace None with a number to limit the display
pd.set_option('display.max_columns', None)  # Replace None with a number to limit the display
pd.set_option('display.width', None)  # Adjust the width for better readability if necessary
pd.set_option('display.max_colwidth', None)  # Adjust column width to show all data in columns

# sorted_df = df.sort_values(by=['tag','extra'])
# print(sorted_df)
df_unique = df.drop_duplicates(subset=['object_id'], keep='first')
print(df_unique)
