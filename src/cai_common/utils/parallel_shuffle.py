import argparse
import random


def main(args):     # pylint: disable=redefined-outer-name
    # Shuffle two files in parallel. Useful for parallel training data preparation.
    with open(args.input_file + '.bo', 'r', encoding='utf-8') as f_bo, open(args.input_file + '.en', 'r') as f_en:
        bo_lines, en_lines = f_bo.readlines(), f_en.readlines()
    combined = list(zip(bo_lines, en_lines))

    random.shuffle(combined)
    if args.num_lines > 0:
        combined = combined[:args.num_lines]

    with open(args.output_file + '.bo', 'a', encoding='utf-8') as f_bo, open(args.output_file + '.en', 'a') as f_en:
        for bo_line, en_line in combined:
            f_bo.write(bo_line)
            f_en.write(en_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input-file',
        type=str,
        help='Input filename, without the extension')
    parser.add_argument(
        '-o',
        '--output-file',
        type=str,
        help='Output filename, without the extension')
    parser.add_argument(
        '-n',
        '--num-lines',
        type=int,
        default=-1,
        help='Number of lines to output')
    parser.add_argument(
        '--seed',
        type=int,
        default=12345,
        help='Random seed')
    args = parser.parse_args()

    main(args)
