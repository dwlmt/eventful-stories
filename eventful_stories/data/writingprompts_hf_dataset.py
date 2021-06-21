import datasets
import os
from datasets.info import SupervisedKeysData
from jsonlines import jsonlines
from typing import Optional

from eventful_stories.data.hf_processing_utils import CorefEventExamples

_VERSION = datasets.Version("1.0.0")

_CITATION = """\
@inproceedings{fan-etal-2018-hierarchical,
    title = "Hierarchical Neural Story Generation",
    author = "Fan, Angela  and
      Lewis, Mike  and
      Dauphin, Yann",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P18-1082",
    doi = "10.18653/v1/P18-1082",
    pages = "889--898",
}
"""

_DESCRIPTION = """\
 The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
 Each story is a creative writing exercise following a prompt.
"""

_URL = "https://drive.google.com/uc?export=download&id=1b8Q4_t2D0IKUXmlTY8sNV5ga0cB8kdvp"

_DOWNLOAD_NUM_BYTES = 250330124
_DOWNLOAD_CHECKSUM = "1c8886bec4948a77d16255f6e80178ae65dc4d28c24f543d5b2f6c7aaa057238"


class WritingPromptsInterleavedHfDatasetConfig(datasets.BuilderConfig):

    def __init__(self,
                 data_url: str,
                 data_download_num_bytes: Optional[int],
                 data_download_checksum: Optional[str],
                 dummy: bool = False,
                 **kwargs):
        """ Generic config for reading a dataset in a interleaved or round robin fashion.

        Args:
            data_url (str): The url for the compressed jsonl file.
            data_download_num_bytes (int): Number of bytes of the datafile.
            data_download_checksum (str): SHA-256 checksum for the data file.
            dummy (bool): If true then only yield the first 10000 examples.
            **kwargs: Pass to parent.
        """
        self.data_url = data_url
        self.data_download_num_bytes = data_download_num_bytes
        self.data_download_checksum = data_download_checksum
        self.dummy = dummy

        super(WritingPromptsInterleavedHfDatasetConfig, self).__init__(**kwargs)


class WritingPromptsInterleavedDataset(datasets.GeneratorBasedBuilder):
    """The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
        Each story is a creative writing exercise following a prompt.
    """

    BUILDER_CONFIG_CLASS = WritingPromptsInterleavedHfDatasetConfig
    BUILDER_CONFIGS = [
        WritingPromptsInterleavedHfDatasetConfig(name="writingprompts_dummy",
                                                 description="Writing Prompts dummy for testng purposes.",
                                                 data_url=_URL,
                                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                                 dummy=True,
                                                 version=_VERSION),
        WritingPromptsInterleavedHfDatasetConfig(name="writingprompts",
                                                 description="Writing Prompts dummy for testng purposes.",
                                                 data_url=_URL,
                                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                                 dummy=True,
                                                 version=_VERSION),

    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),  # The unique id for the story
                    "title": datasets.Value("string"),  # The title of the work.
                    # "text": datasets.Value("string"),  # The context input_text field.
                    "sentences": [datasets.Value("string")],
                    "sentences_corefs": [datasets.Value("string")],
                    "events": [
                        dict([("seq_num", datasets.Value("int32")), ("verb", datasets.Value("string")),
                              ("text", datasets.Value("string"))])],
                    "coreference_glosses": [dict([("id", datasets.Value("string")), ("gloss", datasets.Value("string")),
                                                  ("count", datasets.Value("int32"))])],
                    "coreference_types": [
                        dict([("id", datasets.Value("string")), ("type", datasets.Value("string"))])],
                    "coreference_mentions": [
                        dict([("id", datasets.Value("string")), ("seq_num", datasets.Value("int32"))])]
                }
            ),
            version=_VERSION,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns splits from train,valid,test.jsonl """

        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "WritingPrompts")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "test.jsonl"), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "valid.jsonl"),
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields an example for each story split by stories.
            The prompt is the title but also prepended to the main input_text.
        """
        process = CorefEventExamples()

        with jsonlines.open(filepath, mode='r') as reader:
            for example in process.process_examples(reader,
                                                    dummy=self.config.dummy):
                yield example['id'], example
