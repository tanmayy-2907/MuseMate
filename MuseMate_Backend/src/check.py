import pandas as pd
import os

# --- CONFIGURATION ---
CSV_PATH = 'data/CSV Files/master_artwork_info.csv'
# ---------------------

def verify_artwork(df, filename):
    """Searches the DataFrame for a filename and prints its details."""
    
    # Ensure the filename has the correct extension if the user forgets it
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        filename += '.jpg'

    # Find the row that matches the filename
    artwork_info = df[df['filename'] == filename]

    # Check if the file was found and print the result
    if not artwork_info.empty:
        # Get the first result for that filename
        true_title = artwork_info['title'].iloc[0]
        true_artist = artwork_info['artist'].iloc[0]
        
        print("\n" + "="*40)
        print(f"VERIFICATION FOR: {filename}")
        print("="*40)
        print(f"  > True Title: {true_title}")
        print(f"  > True Artist: {true_artist}")
        print("="*40 + "\n")
    else:
        print(f"\n---> Filename '{filename}' was not found in the dataset.\n")

def main():
    """Main function to load data and run the interactive loop."""
    try:
        print("Loading artwork dataset...")
        df = pd.read_csv(CSV_PATH)
        print("Dataset loaded. You can now check filenames.")
        
        while True:
            # Prompt the user for input
            filename_to_check = input("Enter the filename to check (e.g., 68.jpg) or type 'exit' to quit: ")
            
            # Allow the user to exit the loop
            if filename_to_check.lower() in ['exit', 'quit']:
                break
            
            # Call the verification function
            verify_artwork(df, filename_to_check)

    except FileNotFoundError:
        print(f"Error: The file '{CSV_PATH}' was not found. Make sure you are running this script from the project's root directory.")

# Run the main function
if __name__ == '__main__':
    main()
