import pandas as pd
import argparse


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Merge two CSV files on FileName column"
    )
    parser.add_argument("--csv1", help="Path to the first CSV file")
    parser.add_argument("--csv2", help="Path to the second CSV file")
    parser.add_argument("--output", help="Path to the output merged CSV file")

    # Parse arguments
    args = parser.parse_args()

    # Read the CSV files
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    # Merge on 'FileName'
    merged_df = pd.merge(df1, df2, on="FileName", how="left")

    # Fill missing values with default values
    merged_df['Start_ED'] = merged_df['Start_ED'].fillna(0)
    merged_df['ES'] = merged_df['ES'].fillna(31)
    merged_df['End_ED'] = merged_df['End_ED'].fillna(63)

    # Convert to integers if needed
    merged_df['Start_ED'] = merged_df['Start_ED'].astype(int)
    merged_df['ES'] = merged_df['ES'].astype(int)
    merged_df['End_ED'] = merged_df['End_ED'].astype(int)

    # Save the merged DataFrame
    merged_df.to_csv(args.output, index=False)
    print(f"Merged CSV saved to: {args.output}")


if __name__ == "__main__":
    main()