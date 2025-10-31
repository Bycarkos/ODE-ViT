import torch  # type: ignore
import numpy as np
from typing import Optional


class Collator:
    def __init__(self, processor, pad_token: Optional[int] = -100):
        self.processor = processor
        self._pad_token = pad_token

    def classification_collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        images = [item[0] for item in batch]
        # Process images in one batch operation
        processed_pixels = self.processor(pixel_values, return_tensors="pt")

        return {
            "pixel_values": processed_pixels,
            "labels": torch.tensor(labels),
            "raw_images": images,
        }

    def detector_collate_fn(self, batch):
        pixel_values = [item[0]["image"].unsqueeze(0) for item in batch]
        pixel_values = torch.cat(pixel_values, dim=0)

        encoding = self.processor.pad(pixel_values, return_tensors="pt")

        targets = [
            {
                "labels": torch.from_numpy(np.array(item[0]["labels"])),
                "boxes": torch.from_numpy(np.array(item[0]["bboxes"])),
            }
            for item in batch
        ]

        metadata = [item[-1] for item in batch]

        images = [item[1] for item in batch]

        return dict(
            pixel_values=encoding["pixel_values"],
            pixel_mask=encoding["pixel_mask"],
            targets=targets,
            metadata=metadata,
            images=images,
        )

    def ocr_collate_fn(self, batch):
        """
        Collate function for OCR dataset to handle image and text batching.

        Args:
            batch: List of dictionaries containing samples

        Returns:
            Dictionary of batched tensors and metadata
        """
        # Extract values from batch
        pixel_values = [item["pixel_values"].permute(2, 0, 1) for item in batch]
        text = [item["text"] for item in batch]

        tokens = [item.get("tokens", [])["input_ids"] for item in batch]
        images = [
            item.get("image", None) for item in batch
        ]  # Assuming raw_images should be from "image" key

        # Store original shapes before processing
        original_shapes = [value.shape for value in pixel_values]

        # Process images in one batch operation
        processed_pixels = self.processor(pixel_values, return_tensors="pt")

        # Get max sequence length for efficient padding
        max_seq_len = max(len(t) for t in tokens) if tokens else 0

        # Create padded token tensor efficiently
        if max_seq_len > 0:
            # Initialize with pad token
            padded_tokens = torch.full(
                size=(len(batch), max_seq_len),
                fill_value=self._pad_token,
                dtype=torch.long,  # Specify dtype explicitly
            )

            # Fill in actual token values
            for idx, seq in enumerate(tokens):
                if len(seq) > 0:  # Check if sequence is not empty
                    padded_tokens[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        else:
            # Handle case with empty tokens
            padded_tokens = torch.zeros((len(batch), 0), dtype=torch.long)

        # Store processed shapes for reference
        processed_shapes = [p_val.shape for p_val in processed_pixels["pixel_values"]]

        return {
            "pixel_values": processed_pixels,
            "text": text,
            "tokens": padded_tokens,
            "original_shapes": original_shapes,
            "resized_shapes": processed_shapes,
            "unpadded_tokens": [torch.tensor(t, dtype=torch.long) for t in tokens],
            "raw_images": images,
        }
