import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .processors import NaiveAudioProcessor, FbankAudioProcessor
from ..utils import load_json

def label2caption(label, background_sound=None, template="{} can be heard"):
    r"""This is a helper function converting list of labels to captions."""
    if background_sound is None:
        return [template.format(", ".join(l)) for l in label]

    if isinstance(background_sound, str):
        background_sound = [[background_sound]] * len(label)

    assert len(label) == len(
        background_sound
    ), "the number of `background_sound` should match the number of `label`."

    caption = []
    for l, bg in zip(label, background_sound):
        cap = template.format(", ".join(l))
        cap += " with the background sounds of {}".format(", ".join(bg))
        caption.append(cap)

    return caption


class AudioDataset(Dataset):
    def __init__(
        self,
        metadata_root: str = "/path/to/dataset_root.json",
        dataset_name: list = ["audioset"],
        split: str = "train",
        include_caption: bool = True,
        enable_mixup: bool = False,
        audio_processor: NaiveAudioProcessor = NaiveAudioProcessor(),
    ):
        """
        Dataset that manages audio recordings.
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.metadata_root = load_json(metadata_root)
        self.dataset_name = dataset_name
        self.split = split
        self.include_caption = include_caption
        self.audio_processor = audio_processor

        self.enable_mixup = enable_mixup
        self.mixture_caption_template = "{} | {}"
        if self.enable_mixup:
            print(
                f"Template for the caption of mixture is: {self.mixture_caption_template}"
            )

        self.build_dataset()
        print("Dataset initialization finished.")

    def __getitem__(self, index):
        datum = self.data[index]
        fname = datum["wav"]  # base name of the wav file

        mix_datum = {"wav": None}
        if self.enable_mixup:
            if random.random() > 0.5:
                mix_datum = self.data[random.randint(0, len(self.data) - 1)]
                fname += " " + mix_datum["wav"]

        data = {"fname": fname}

        if self.include_caption:
            caption = self.get_caption_from_datum(
                datum,
                mix_datum,
                template_description=self.mixture_caption_template,
            )
            data.update({"caption": caption})

        data.update(self.audio_processor(datum["wav"], mix_datum["wav"]))

        return data

    def text_to_filename(self, text):
        return text.replace(" ", "_").replace("'", "_").replace('"', "_")

    def get_dataset_root_path(self, dataset):
        assert dataset in self.metadata_root.keys()
        return self.metadata_root[dataset]

    def get_dataset_metadata_path(self, dataset, key):
        # key: train, test, val, class_label_indices
        try:
            if dataset in self.metadata_root["metadata"]["path"].keys():
                return self.metadata_root["metadata"]["path"][dataset][key]
        except:
            raise ValueError(
                'Dataset %s does not metadata "%s" specified' % (dataset, key)
            )

    def __len__(self):
        return len(self.data)

    def _relative_path_to_absolute_path(self, metadata, dataset_name):
        root_path = self.get_dataset_root_path(dataset_name)
        for i in range(len(metadata["data"])):
            assert "wav" in metadata["data"][i].keys(), metadata["data"][i]
            assert metadata["data"][i]["wav"][0] != "/", (
                "The dataset metadata should only contain relative path to the audio file: "
                + str(metadata["data"][i]["wav"])
            )
            metadata["data"][i]["wav"] = os.path.join(
                root_path, metadata["data"][i]["wav"]
            )
        return metadata

    def build_dataset(self):
        self.data = []
        print("Build dataset split %s from %s" % (self.split, self.dataset_name))
        if type(self.dataset_name) is str:
            data_json = load_json(
                self.get_dataset_metadata_path(self.dataset_name, key=self.split)
            )
            data_json = self._relative_path_to_absolute_path(
                data_json, self.dataset_name
            )
            self.data = data_json["data"]
        elif type(self.dataset_name) is list:
            for dataset_name in self.dataset_name:
                data_json = load_json(
                    self.get_dataset_metadata_path(dataset_name, key=self.split)
                )
                data_json = self._relative_path_to_absolute_path(
                    data_json, dataset_name
                )
                self.data += data_json["data"]
        else:
            raise Exception("Invalid data format")
        print("Data size: {}".format(len(self.data)))

    def is_contain_caption(self, datum):
        if datum is not None:
            caption_keys = [x for x in datum.keys() if ("caption" in x)]
            return len(caption_keys) > 0
        else:
            return False

    def _read_datum_caption(self, datum):
        if datum is not None:
            caption_keys = [x for x in datum.keys() if ("caption" in x)]
            random_index = torch.randint(0, len(caption_keys), (1,))[0].item()
            return datum[caption_keys[random_index]]
        else:
            return ""  # NOTE: return empty string if datum is not provided

    def label_indices_to_text(
        self,
        datum,
        label_indices,
        template_description: str = "{}",  # e.g., "This audio contains the sound of {}"
    ):
        if self.is_contain_caption(datum):
            return self._read_datum_caption(datum)

        elif "label" in datum.keys():
            name_indices = torch.where(label_indices > 0.1)[0]

            labels = ""
            for id, each in enumerate(name_indices):
                if id == len(name_indices) - 1:
                    labels += "%s." % self.num2label[int(each)]
                else:
                    labels += "%s, " % self.num2label[int(each)]
            return template_description.format(labels)

        else:
            return ""  # NOTE: return empty string if both label and caption are not provided

    def get_sample_text_caption(self, datum, mix_datum, label_indices):
        text = self.label_indices_to_text(datum, label_indices)
        if mix_datum is not None:
            text += " " + self.label_indices_to_text(mix_datum, label_indices)
        return text

    def get_caption_from_datum(
        self, datum, mix_datum=None, template_description="{} {}"
    ):
        caption = ""
        if self.is_contain_caption(datum):
            caption += self._read_datum_caption(datum)

        # Mixup the caption if `mix_datum` is not None
        if mix_datum is not None and self.is_contain_caption(mix_datum):
            mix_caption = self._read_datum_caption(mix_datum)
            caption = template_description.format(caption, mix_caption)

        return caption


if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataset = AudioDataset(
        dataset_name=["audiocaps"],
        include_caption=True,
        enable_mixup=True,
        audio_processor=FbankAudioProcessor(),
    )

    loader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True)

    for cnt, each in tqdm(enumerate(loader)):
        # print(each["waveform"].size(), each["log_mel_spec"].size())
        # print(each['freq_energy_percentile'])
        import ipdb; ipdb.set_trace()
