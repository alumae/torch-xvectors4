
from collections import OrderedDict
import logging
import sys
import argparse

import kaldiio
import torch
from tqdm import tqdm
import numpy as np

from model import SpeechClassificationModel

def load_simple_scp(wav_scp):
    for l in open(wav_scp):
        
        uttid, audio_path = l.split(maxsplit=1)
        #print(audio_path)
        audio_path = audio_path.strip()
        audio_path = audio_path.replace("wav-copy", "wav-copy --print-args=false")
        utt_sample_rate, sound_np = kaldiio.load_mat(audio_path)
        yield uttid, (utt_sample_rate, sound_np)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    parser = argparse.ArgumentParser()  
    parser.add_argument("model", type=str)  
    parser.add_argument("--segments", type=str)  
    parser.add_argument("--batch-size", default=256, type=int)  
    parser.add_argument("--device", default="cuda", type=str)  
    parser.add_argument("--max-segment-length", type=float, default=0.0)      
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

        for key, (rate, numpy_array)  in tqdm(kaldiio.load_scp_sequential(args.wav_scp, segments=args.segments)):
            #for in tqdm(reader):
            audio = torch.FloatTensor(numpy_array)
            audio = audio / 2**15
            if args.max_segment_length > 0.0:
                max_in_samples = int(args.max_segment_length * 16000)
                audio = audio[0:max_in_samples]            
            audio_batch.append(audio)
            keys_batch.append(key)

            # Check if the batch is full
            if len(audio_batch) == args.batch_size:
                # Convert list of tensors to a single tensor
                wav_lens = torch.tensor([len(audio) for audio in audio_batch]).float()
                wav_lens /= wav_lens.max()
                audio_tensor = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True, padding_value=0.0).to(device)
                #breakpoint()
                # Process the batch through the model
                em_batch = model.extract_xvectors(audio_tensor, wav_lens=wav_lens).detach().cpu()

                # Write the results
                for i, key in enumerate(keys_batch):
                    write_helper(key, em_batch[i].cpu().numpy().flatten())

                # Clear the batch
                audio_batch = []
                keys_batch = []

        # Handle the last batch
        if audio_batch:
            wav_lens = torch.tensor([len(audio) for audio in audio_batch]).float()
            wav_lens /= wav_lens.max()
            audio_tensor = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True, padding_value=0.0).to(device)
            em_batch = model.extract_xvectors(audio_tensor, wav_lens=wav_lens).detach().cpu()
            for i, key in enumerate(keys_batch):
                write_helper(key, em_batch[i].cpu().numpy().flatten())

    # Don't forget to close the write_helper
    write_helper.close()
