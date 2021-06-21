from __future__ import absolute_import, division, print_function

import datasets
import glob
import os
import pathlib
from datasets.info import SupervisedKeysData
from jsonlines import jsonlines
from random import Random

from eventful_stories.data.hf_processing_utils import CorefEventExamples

_DESCRIPTION = """\
Wrapper for multiple interleaving versions of a glob corpus.
"""

_VERSION = datasets.Version("1.0.0")

_CITATION = """\

"""
_PROJECT_URL = ""

_BOOK_CORPUS_URL = "https://t.co/J3EaSEgwW0?amp=1"
_BOOK_CORPUS_GLOB_PATH = "**/*.epub.txt"

_SCHMOOP_CORPUS_URL = "https://drive.google.com/uc?export=download&id=1y5Ac3LARFuMAV0bmxy9V91JFkc8J59nP"
_SCHMOOP_CORPUS_GLOB_PATH = "**/*.txt.utf8"

_MOVIE_CORPUS_URL = "https://drive.google.com/uc?export=download&id=16DBMpLY-w5ZF0yph-D3lhRjS_Cgwj-vZ"
_MOVIE_CORPUS_GLOB_PATH = "**/scripts/parsed/full/*.txt"

_GUTENBERG_CORPUS_URL = "https://drive.google.com/uc?export=download&id=1dObECu3jGIAFpMQtrkIu2uTwdLFsAvfj"
_GUTENBERG_CORPUS_GLOB_PATH = "**/*.txt"


class GlobInterleavedHfDatasetConfig(datasets.BuilderConfig):

    def __init__(self,
                 data_url: str,
                 glob_path: str,
                 dummy: bool = False,
                 **kwargs):
        """ Generic config for reading a dataset in a interleaved or round robin fashion.

        Args:
            data_url (str): The url for the compressed jsonl file.
            data_download_num_bytes (int): Number of bytes of the datafile.
            data_download_checksum (str): SHA-256 checksum for the data file.
            input_size (int): Size in sentences of the context input_text to condition on.
            target_size (int): Size in sentences of the input_text target_text to predict.
            step_size (int): Sliding window step to pass over the input_text.
            batch_size (int): Number of stories to iterate over in parallel.
            dummy (bool): If true then only yield the first 10000 examples.
            **kwargs: Pass to parent.
        """
        self.data_url = data_url
        self.glob_path = glob_path
        self.dummy = dummy
        super(GlobInterleavedHfDatasetConfig, self).__init__(**kwargs)


class GlobCorpusOpen(datasets.GeneratorBasedBuilder):
    """Wrapper for a GLOB text corpus. """

    BUILDER_CONFIG_CLASS = GlobInterleavedHfDatasetConfig
    BUILDER_CONFIGS = [
        GlobInterleavedHfDatasetConfig(name="bookcorpus_dummy",
                                       description="Bookcorpus dummy dataset.",
                                       data_url=_BOOK_CORPUS_URL,
                                       glob_path=_BOOK_CORPUS_GLOB_PATH,
                                       version=_VERSION,
                                       dummy=True),
        GlobInterleavedHfDatasetConfig(name="bookcorpus",
                                       description="Bookcorpus full dataset.",
                                       data_url=_BOOK_CORPUS_URL,
                                       glob_path=_BOOK_CORPUS_GLOB_PATH,
                                       version=_VERSION,
                                       dummy=False),
        GlobInterleavedHfDatasetConfig(name="schmoop_dummy",
                                       description="Schmoop dummy for testing purposes.",
                                       data_url=_SCHMOOP_CORPUS_URL,
                                       glob_path=_SCHMOOP_CORPUS_GLOB_PATH,
                                       dummy=True,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="schmoop",
                                       description="Schmoop full dataset.",
                                       data_url=_SCHMOOP_CORPUS_URL,
                                       glob_path=_SCHMOOP_CORPUS_GLOB_PATH,
                                       dummy=False,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="moviecorpus_dummy",
                                       description="Movie script dummy for testing purposes.",
                                       data_url=_MOVIE_CORPUS_URL,
                                       glob_path=_MOVIE_CORPUS_GLOB_PATH,
                                       dummy=True,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="moviecorpus",
                                       description="Movie full dataset.",
                                       data_url=_MOVIE_CORPUS_URL,
                                       glob_path=_MOVIE_CORPUS_GLOB_PATH,
                                       dummy=False,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="gutenberg_dummy",
                                       description="Gutenberg dummy corpus",
                                       dummy=True,
                                       data_url=_GUTENBERG_CORPUS_URL,
                                       glob_path=_GUTENBERG_CORPUS_GLOB_PATH,
                                       version=_VERSION),
        GlobInterleavedHfDatasetConfig(name="gutenberg",
                                       description="Gutenberg full corpus",
                                       dummy=True,
                                       data_url=_GUTENBERG_CORPUS_URL,
                                       glob_path=_GUTENBERG_CORPUS_GLOB_PATH,
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
        arch_path = dl_manager.download_and_extract(self.config.data_url)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": arch_path}),
        ]

    def _generate_examples(self, filepath):
        """ Yields an example for each story split by stories.
            The prompt is the title but also prepended to the main input_text.
        """
        # bbb

        process = CorefEventExamples()

        glob_target = os.path.join(filepath, self.config.glob_path)
        book_files = glob.glob(glob_target, recursive=True)

        def _reader(book_files):
            _id = 0
            for book_file_path in book_files:
                path = pathlib.PurePath(book_file_path)
                try:
                    with open(book_file_path, mode="r", encoding="utf-8") as f:
                        glob_dict = {"title": str(path.name), "text": f.read(), "id": str(path.name)}
                        yield glob_dict
                        _id += 1
                except Exception as e:
                    print(f"{e}")

        for example in process.process_examples(_reader(book_files=book_files),
                                                dummy=self.config.dummy):
            yield example['id'], example
