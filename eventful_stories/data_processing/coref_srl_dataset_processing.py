''' Script for cluster analysis for story vectors.
'''

import fire

class ProcessSRLDataset(object):
    ''' Convert the ProppLearner XML file to json that can be evaluated against using the RAG model.
    '''

    def process(self, data_file: str, output_file: str, script_path: str = "glob_hf_dataset.py", dataset_name: str = "bookscorpus"):

        script_path = f"../data/{script_path}"
        from datasets import load_dataset

        dataset = load_dataset(script_path, name=dataset_name,
                               split='train', data_files=[data_file])

        for d in dataset:
            print(f"EXAMPLE: {d}")

        dataset.save_to_disk(output_file)

if __name__ == '__main__':
    fire.Fire(ProcessSRLDataset)
