# type: ignore
## geometrical data packages
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from transformers.models.trocr import TrOCRProcessor


## commmon packages
import os
import glob
from PIL import Image
import numpy as np


class EsposallesDatasetForHtr(Dataset):

    def __init__(self, dataset_path, transforms, tokenizer, type_image: str = "words", add_special_tokens:bool = False):

        self._dataset_path = dataset_path
        self._images_paths = []
        self._transcriptions = []
        self.transforms = transforms
        self._tokenizer = tokenizer

#        self._blank_token_id = tokenizer.blank_token_id#("_")["input_ids"][1]
        self.blank_token_id = tokenizer.blank_token_id
        self._pad_token_id = self._tokenizer.pad_token_id
        self._add_special_tokens = add_special_tokens
        self._nb_classes = len(self._tokenizer)

        for top_folder, folder_name in enumerate(
            sorted(glob.glob(self._dataset_path + "/**/"))
        ):
            page_records_folder = sorted(glob.glob(folder_name + "/**/"))
            for idPageRecord in page_records_folder:
                images_path = os.path.join(idPageRecord, type_image)
                basename = os.path.basename(os.path.split(idPageRecord)[0])
                txt_transcriptions_path = os.path.join(
                    images_path, basename + "_transcription.txt"
                )

                with open(txt_transcriptions_path, "r") as file:
                    transcriptions = file.readlines()
                images_path, transcriptions = list(
                    zip(
                        *[
                            (
                                os.path.join(images_path, im.split(":")[0]),
                                im.split(":")[1].strip(),
                            )
                            for im in transcriptions
                        ]
                    )
                )

                self._transcriptions += transcriptions
                self._images_paths += images_path

    def __len__(self):
        return len(self._images_paths)

    @property
    def tokenizer(self):
        return self._tokenizer

    def __getitem__(self, idx):

        pixel_values = Image.open(self._images_paths[idx] + ".png").convert("RGB")
        text = self._transcriptions[idx]

        tokenization = self._tokenizer(text.strip(), add_special_tokens=self._add_special_tokens)

        array_image = np.array(pixel_values)

        return dict(
            pixel_values=torch.from_numpy(array_image),
            text=text,
            tokens=tokenization,
            image=array_image,
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer,AutoImageProcessor, ViTMAEForPreTraining

    transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained("google/canine-s")


    espo = EsposallesDatasetForHtr(
        dataset_path="/data/users/cboned/data/HTR/Esposalles",
        transforms=transforms,
        tokenizer=tokenizer,
    )

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

    import pdb
    pdb.set_trace()
    # dloader = DataLoader(espo, batch_size=2, collate_fn=espo.collate_fn)
    espo[0]
