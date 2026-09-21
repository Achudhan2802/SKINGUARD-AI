import os
import glob

# Search common locations on the C drive
search_term_images = "HAM10000_images"
search_term_csv = "HAM10000_metadata.csv"

print("Searching your computer for the dataset files...")

found_images = glob.glob(r"C:\**\\" + search_term_images, recursive=True)
found_csv = glob.glob(r"C:\**\\" + search_term_csv, recursive=True)

print("\n--- RESULTS ---")
print("Images paths found:", found_images)
print("CSV paths found:", found_csv)