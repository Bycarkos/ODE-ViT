from typing import Any, Dict, List, cast
import numpy as np
import pdb
from typing import *

class GreedyTextDecoder:
    # Pau's implementation:
    # https://github.com/ptorras/comref-experiments/blob/master/src/core/formatters/ctc/greedy_decoder.py
    """Generate an unpadded token sequence from a CTC output."""

    def __init__(self,blank_index: int, tokenizer, confidences: bool = False) -> None:
        """Construct GreedyTextDecoder object."""
        super().__init__()
        self._confidences = confidences
        self.tokenizer = tokenizer
        self._blank_index = blank_index

    
    
    
    def __call__(
        self, ctc_output, *args
    ) -> List[Dict[str, Any]]:
        """Convert a model output to a token sequence.

        Parameters
        ----------
        model_output: ModelOutput
            The output of a CTC model. Should contain an output with shape B x L x C,
            where L is the sequence length, B is the batch size and C is the number of
            classes.
        batch: BatchedSample
            Batch information.

        Returns
        -------
        List[Dict[str, Any]]
            A List of sequences of tokens corresponding to the decoded output and the
            output confidences encapsulated within a dictionary.
        """
        indices = ctc_output.argmax(axis=-1)
        output = []

        for sample, mat in zip(indices, ctc_output):
            previous = self._blank_index
            decoded = []
            confs = []
            for ind, element in enumerate(sample):
                if element == self._blank_index:
                    previous = self._blank_index
                    continue
                if element == previous:
                    continue
                decoded.append(element)
                previous = element
                confs.append(mat[ind])

            decoded = np.array(decoded)
            confs = np.array(confs)
            if self._confidences:
                output.append({"text": decoded, "text_conf": confs})
            else:
                output.append({"text": decoded, "text_conf": None})

        return output