import json
import os
import datasets

_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"

class MathQa(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="MathQA Dataset",
            features=datasets.Features(
                {
                    "Problem": datasets.Value("string"),
                    "Rationale": datasets.Value("string"),
                    "options": datasets.Value("string"),
                    "correct": datasets.Value("string"),
                    "annotated_formula": datasets.Value("string"),
                    "linear_formula": datasets.Value("string"),
                    "category": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://math-qa.github.io/math-QA/",
            citation="None for now",
        )

    def _split_generators(self, dl_manager):
        dl_path = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(dl_path, "train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(dl_path, "test.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(dl_path, "dev.json")},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for id_, row in enumerate(data):
                yield id_, row
