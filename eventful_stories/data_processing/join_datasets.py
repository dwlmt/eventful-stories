''' Use GLOB to join the datasets..
'''
import glob

import fire
import os
from pathlib import Path
from datasets import concatenate_datasets, load_dataset, load_from_disk


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class JoinDataset(object):
    ''' Use GLOB path to join the datasets together.
    '''

    def join(self, input_path, glob_path, output_path):

        ensure_dir(output_path)

        print(f"Input Path: {input_path}, Input GLOB path: {glob_path}, Output path: {output_path}")

        paths = Path(input_path).glob(glob_path)
        datasets = []
        for path in paths:
            print(f"Path: {path}")

            d = load_from_disk(path)
            #print(d)
            datasets.append(d)

        joined_dataset = concatenate_datasets(datasets)

        joined_dataset.save_to_disk(output_path)
        print(f"Dataset: {joined_dataset.info}")

        #for d in joined_dataset:
        #    print(f"Book: {d}")


if __name__ == '__main__':
    fire.Fire(JoinDataset)