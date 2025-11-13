#!/usr/bin/env python3

import random
import argparse
import sys
from pathlib import Path

# --- THIS IS THE NEW LINE ---
# By setting a specific seed, the "random" sampling will be the same
# every time the script is run. You can change 42 to any other integer.
random.seed(42)

def sample_lines_from_file(filepath: Path, percentage: int):
    """
    Reads a text file, randomly samples a specified percentage of its lines,
    and saves them to a new file.
    """
    
    # 1. Read all lines from the source file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    if not lines:
        print(f"The file {filepath} is empty. No output file created.")
        return

    # 2. Calculate the number of lines to sample
    total_lines = len(lines)
    num_to_sample = round(total_lines * (percentage / 100))
    
    if num_to_sample == 0 and total_lines > 0:
        num_to_sample = 1
        
    # 3. Use random.sample() to pick unique lines
    # This will now be reproducible thanks to random.seed(42)
    sampled_lines = random.sample(lines, num_to_sample)
    
    # 4. Construct the output filename
    output_filename = f"{filepath.stem}_{percentage}rand{filepath.suffix}"
    output_filepath = filepath.with_name(output_filename)
    
    # 5. Write the sampled lines to the new file
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.writelines(sampled_lines)
        
        print(f"Success! Saved {num_to_sample} reproducible random lines ({percentage}%) to:")
        print(f"{output_filepath}")
        
    except Exception as e:
        print(f"Error writing to output file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to parse command-line arguments.
    """
    
    def percentage_type(x):
        try:
            x_int = int(x)
            if not 1 <= x_int <= 99:
                raise argparse.ArgumentTypeError("Percentage must be an integer between 1 and 99.")
            return x_int
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{x}' is not a valid integer.")

    parser = argparse.ArgumentParser(
        description="Randomly sample a percentage of lines from a .txt file.",
        epilog="Example: python sample_lines.py 80 my_data.txt"
    )
    
    parser.add_argument(
        'percentage',
        type=percentage_type,
        help="The percentage of lines to sample (an integer from 1 to 99)."
    )
    
    parser.add_argument(
        'filepath',
        type=Path,
        help="The path to the input .txt file."
    )
    
    args = parser.parse_args()
    
    # Run the main function
    sample_lines_from_file(args.filepath, args.percentage)

if __name__ == "__main__":
    main()