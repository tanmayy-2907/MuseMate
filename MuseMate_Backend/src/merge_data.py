import pandas as pd

df_train = pd.read_csv('data/CSV Files/train_info.csv')
df_all_data = pd.read_csv('data/CSV Files/all_data_info.csv')


df_merged = pd.merge(df_train, df_all_data, 
                     left_on='filename', 
                     right_on='new_filename', 
                     how='left')

# 3. Clean up the final DataFrame (Recommended)
# The merge creates duplicate columns (e.g., artist_x, artist_y). Let's clean it up.
# We will keep the filename, the readable artist name, and other useful info.
df_master = df_merged[[
    'filename', 
    'artist_y', # This is the readable artist name from all_data_info.csv
    'title_x', 
    'style_x', 
    'genre_x', 
    'date_x'
]].copy() # Using .copy() to avoid a SettingWithCopyWarning

# Rename the columns for clarity
df_master.columns = ['filename', 'artist', 'title', 'style', 'genre', 'date']


print("Successfully created the master file!")
print(df_master.head())

#Save this new master file to a new CSV
#df_master.to_csv('master_artwork_info.csv', index=False)