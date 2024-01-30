
from collections import OrderedDict
import logging
import sys
import argparse

import kaldiio
import torch
from tqdm import tqdm
import numpy as np

from model import SpeechClassificationModel



if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    parser = argparse.ArgumentParser()  
    parser.add_argument("model", type=str)  
    parser.add_argument("--segments", type=str)  
    parser.add_argument("--batch-size", default=256, type=int)  
    parser.add_argument("--device", default="cuda", type=str)  
    parser.add_argument("wav_scp")    
    parser.add_argument("out_dir")
    args = parser.parse_args()



    device = torch.device(args.device)

    model = SpeechClassificationModel.load_from_checkpoint(args.model).to(device)
    model.eval()

    write_helper = kaldiio.WriteHelper(f'ark,scp:{args.out_dir}/xvector.ark,{args.out_dir}/xvector.scp')

    audio_batch = []
    keys_batch = []

    with torch.no_grad():
        with kaldiio.ReadHelper(f'scp:{args.wav_scp}', segments=args.segments) as reader:
            for key, (rate, numpy_array) in tqdm(reader):
                audio = torch.FloatTensor(numpy_array).to(device)
                audio = audio / 2**15
                audio_batch.append(audio)
                keys_batch.append(key)

                # Check if the batch is full
                if len(audio_batch) == args.batch_size:
                    # Convert list of tensors to a single tensor
                    audio_tensor = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True, padding_value=0.0).to(device)
                    #breakpoint()
                    # Process the batch through the model
                    em_batch = model.extract_xvectors(audio_tensor, wav_lens=None).detach().cpu()

                    # Write the results
                    for i, key in enumerate(keys_batch):
                        write_helper(key, em_batch[i].cpu().numpy().flatten())

                    # Clear the batch
                    audio_batch = []
                    keys_batch = []

            # Handle the last batch
            if audio_batch:
                audio_tensor = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True, padding_value=0.0).to(device)
                em_batch = model.extract_xvectors(audio_tensor, wav_lens=None).detach().cpu()
                for i, key in enumerate(keys_batch):
                    write_helper(key, em_batch[i].cpu().numpy().flatten())

    # Don't forget to close the write_helper
    write_helper.close()
