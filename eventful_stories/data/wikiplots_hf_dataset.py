import datasets
from datasets.info import SupervisedKeysData
from jsonlines import jsonlines
from typing import Optional

from eventful_stories.data.hf_processing_utils import CorefEventExamples

_CITATION = ""

_DESCRIPTION = """\
 English language plots taken from the English Wikipedia from films, books, plays and other narrative forms. The dataset
 has 132,358 plots in total.
"""

_VERSION = datasets.Version("1.0.0")

_URL = "https://drive.google.com/uc?export=download&id=1PUsmqVzB8SRIRFkBrHAJCL3guojnCjh7"

_HOMEPAGE = "https://github.com/markriedl/WikiPlots"

_DOWNLOAD_NUM_BYTES = 109300457
_DOWNLOAD_CHECKSUM = "7fe76225dcff4ff53830f7272d298a9c2f57e091f76411c652db7b2fed04ed78"


class WikiPlotsHfDatasetConfig(datasets.BuilderConfig):

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
            **kwargs: Pass to parent.
        """
        self.data_url = data_url
        self.data_download_num_bytes = data_download_num_bytes
        self.data_download_checksum = data_download_checksum
        self.dummy = dummy

        super(WikiPlotsHfDatasetConfig, self).__init__(**kwargs)


class WikiplotsDataset(datasets.GeneratorBasedBuilder):
    """ English language plots taken from the English Wikipedia from films, books, plays and other narrative forms. The dataset
            has 132,358 plots in total.
    """

    BUILDER_CONFIG_CLASS = WikiPlotsHfDatasetConfig
    BUILDER_CONFIGS = [
        WikiPlotsHfDatasetConfig(name="wikiplots_dummy",
                                 description="Wikiplots dummy smaller dataset.",
                                 data_url=_URL,
                                 data_download_num_bytes=_DOWNLOAD_NUM_BYTES,
                                 data_download_checksum=_DOWNLOAD_CHECKSUM,
                                 version=_VERSION,
                                 dummy=True)
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
        """Train only as will use datasets functionality to split dynamically."""

        dl_file = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": dl_file
                },
            )
        ]

    def _generate_examples(self, filepath):
        """ Yields an example for each story split by stories.
            The prompt is the title but also prepended to the main input_text.
        """
        # bbb

        process = CorefEventExamples()

        with jsonlines.open(filepath, mode='r') as reader:
            for example in process.process_examples(reader,
                                                    dummy=self.config.dummy):
                yield example['id'], example
