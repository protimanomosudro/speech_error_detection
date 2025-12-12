"""
create_contrive_set.py

This script creates contrived training, evaluation, and testing CSV files
with a balanced ratio of events to non-events.

Usage:
    python create_contrive_set.py --csv_dir <csv_dir> --output_dir <output_dir> [--ratio <ratio>] [--seed <seed>]

    - csv_dir (str): Directory containing the original train.csv, eval.csv, and test.csv files.
    - output_dir (str): Directory to save the contrived CSV files.
    - ratio (float, optional): Desired ratio of event samples. Default is 0.5.
    - seed (int, optional): Random seed for reproducibility. Default is 42.

Example:
    python create_contrive_set.py --csv_dir /data/metadata --output_dir /data/contrived --ratio 0.5 --seed 42
"""

import os
import csv
import random
import argparse
from typing import List, Tuple


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create contrived train, eval, and test CSV files with balanced event and non-event samples."
    )
    parser.add_argument('--csv_dir', type=str, required=True,
                        help="Directory containing the original train.csv, eval.csv, and test.csv files.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the contrived CSV files.")
    parser.add_argument('--ratio', type=float, default=0.5,
                        help="Desired ratio of event samples in the contrived dataset. Default is 0.5.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility. Default is 42.")

    args = parser.parse_args()
    return args


def read_csv(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Read a CSV file and return headers and rows.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Tuple[List[str], List[List[str]]]: Headers and list of rows.
    """
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    return headers, rows


def write_csv(headers: List[str], rows: List[List[str]], file_path: str) -> None:
    """
    Write headers and rows to a CSV file.

    Args:
        headers (List[str]): Header row.
        rows (List[List[str]]): Data rows.
        file_path (str): Path to save the CSV file.
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def separate_events(rows: List[List[str]], label_index: int, events_to_consider: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Separate rows into event and non-event samples.

    Args:
        rows (List[List[str]]): All data rows.
        label_index (int): Index of the label_list column.
        events_to_consider (List[str]): List of event labels to consider.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: Event samples and non-event samples.
    """
    event_samples = []
    non_event_samples = []
    for row in rows:
        label = row[label_index]
        if any(event in label for event in events_to_consider):
            event_samples.append(row)
        else:
            non_event_samples.append(row)
    return event_samples, non_event_samples


def create_contrived_set(
    headers: List[str],
    event_samples: List[List[str]],
    non_event_samples: List[List[str]],
    ratio: float,
    seed: int
) -> List[List[str]]:
    """
    Create a contrived dataset with a specified ratio of event samples.

    Args:
        headers (List[str]): Header row.
        event_samples (List[List[str]]): Event samples.
        non_event_samples (List[List[str]]): Non-event samples.
        ratio (float): Desired ratio of event samples.
        seed (int): Random seed.

    Returns:
        List[List[str]]: Contrived dataset rows.
    """
    event_count = len(event_samples)
    if ratio <= 0 or ratio > 1:
        raise ValueError("Ratio must be between 0 and 1 (exclusive).")

    # Calculate the number of non-event samples needed
    non_event_count = int(event_count * (1 - ratio) / ratio)

    if non_event_count > len(non_event_samples):
        raise ValueError(f"Not enough non-event samples to achieve the desired ratio. "
                         f"Required: {non_event_count}, Available: {len(non_event_samples)}.")

    # Sample non-event samples
    sampled_non_events = random.sample(non_event_samples, non_event_count)

    # Combine event and sampled non-event samples
    contrived_set = event_samples + sampled_non_events

    return contrived_set


def process_split(
    split_name: str,
    csv_dir: str,
    output_dir: str,
    ratio: float,
    seed: int,
    events_to_consider: List[str]
) -> None:
    """
    Process a single data split (train/eval/test) to create a contrived set.

    Args:
        split_name (str): Name of the split ('train', 'eval', 'test').
        csv_dir (str): Directory containing the original CSV files.
        output_dir (str): Directory to save the contrived CSV files.
        ratio (float): Desired ratio of event samples.
        seed (int): Random seed.
        events_to_consider (List[str]): List of event labels to consider.
    """
    input_file = os.path.join(csv_dir, f"{split_name}.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    headers, rows = read_csv(input_file)

    if 'label_list' not in headers:
        raise ValueError(f"'label_list' column not found in {input_file}.")

    label_index = headers.index('label_list')

    event_samples, non_event_samples = separate_events(
        rows, label_index, events_to_consider)

    if not event_samples:
        raise ValueError(f"No event samples found in {split_name}.csv.")

    contrived_set = create_contrived_set(
        headers,
        event_samples,
        non_event_samples,
        ratio,
        seed
    )

    output_file = os.path.join(
        output_dir, f"{split_name}_contrive_{ratio:.2f}.csv")
    write_csv(headers, contrived_set, output_file)

    print(f"Contrived {split_name} set saved to {output_file}. "
          f"Events: {len(event_samples)}, Non-events: {len(contrived_set) - len(event_samples)}, Total: {len(contrived_set)}")


def main():
    args = parse_arguments()

    csv_dir = args.csv_dir
    output_dir = args.output_dir
    ratio = args.ratio
    seed = args.seed

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Define events to consider
    events_to_consider = [
        'Phonological Addition',
        'Phonological Deletion',
        'Phonological Substitution'
    ]

    # Process each split
    for split in ['train', 'eval', 'test']:
        process_split(split, csv_dir, output_dir,
                      ratio, seed, events_to_consider)


if __name__ == "__main__":
    main()
