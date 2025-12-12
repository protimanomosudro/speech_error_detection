#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import re
import json
import torch
import logging
import pandas as pd
import numpy as np

#from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
from evaluate import load
from tqdm import tqdm
from datetime import datetime
import time
from faster_whisper import WhisperModel


start_time = time.time() 

def parse_metadata(df):
    entries = []
    
    for index, row in df.iterrows():
        # Extract values from the DataFrame row
        raw_id = row.get('audioFile', '')  # Adjust column name as needed
        srt_time = row.get('longFormStart', '')  # Adjust column name
        end_time = row.get('longFormEnd', '')  # Adjust column name
        ground_truth = row.get('longFormError', '')  # Adjust column name
        
        # Process the ID to get base and extension
        if raw_id:
            base = os.path.splitext(os.path.basename(str(raw_id)))[0]
            original_extension = os.path.splitext(os.path.basename(str(raw_id)))[1]
        else:
            base, original_extension = None, None
        
        # Process ground truth text
        if ground_truth:
            ground_truth = (
                str(ground_truth)
                .replace(' <COMMA>', ',')
                .replace(' <PERIOD>', '.')
                .replace(' <QUESTIONMARK>', '?')
                .replace(' <EXCLAMATIONPOINT>', '!')
                .lower()
            )
        
        # Only add entry if all required fields are present
        if base and original_extension and srt_time and end_time and ground_truth:
            #print('while parsing', srt_time, end_time, str(srt_time), str(end_time))
            entries.append((base, original_extension, str(srt_time), str(end_time), ground_truth))
    
    return entries

def extract_prefix(audio_id):
    # Extract first 2-3 letters (before any numbers or underscores)
    match = re.match(r'^([a-zA-Z]{2,3})', audio_id)
    if match:
        return match.group(1)
    return None
    
def transcribe_audios(df, audio_dir, save_path):
    # Setup Faster Whisper model
    model_size = "small"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Parse metadata file
    entries = parse_metadata(df)

    all_audio_info = []
    for idx, (base_id, base_ext, srt_time, end_time, ground_truth) in enumerate(tqdm(entries, desc="Processing audios")):
        start_ms = int(float(srt_time) * 1000)
        end_ms = int(float(end_time) * 1000)
        seg_filename = f"{base_id}_{start_ms}_{end_ms}{base_ext}" ## refactor code to consider any type of audio files
        file_path = os.path.join(audio_dir, seg_filename)

        if not os.path.isfile(file_path):
            print(f" Audio file not found for segment: {seg_filename}")
            continue
        
        num_hypotheses = 10
        beam_sizes = [1, 2, 3, 4, 5,6,7,8,9,10][:num_hypotheses]
        inference_texts = []
        # Run inference
        for _, beam_size in enumerate(beam_sizes):
            try: 
                segments, info = model.transcribe(
                file_path,
                beam_size=beam_size,
                language="en",
                vad_filter=True,
                temperature=0.0,
                best_of=10,
                word_timestamps=False  # Enable word-level timestamps
                )
            
                # Collect all segments 
                segments_list = []
                for segment in segments:
                    segment_data = {
                        'text': segment.text,
                    }               
                    segments_list.append(segment_data)
                
                # Combine text for full transcription
                full_text = " ".join([seg['text'] for seg in segments_list])
                
                # Append just the text to the inference_texts list
                inference_texts.append(full_text)
                
                # Create the final output structure
                #result = {
                #    'inference': inference_texts
                #}
                
            except Exception as e:
                print(f"Error with beam size {beam_size}: {e}")
                inference_texts.append("")  # Add empty string if there's an error
                continue
        
        
        # Build metadata dict
        prefix = extract_prefix(base_id) 
        audio_info = {
            'segment_id': str(idx + 1),   # running count
            'ground_truth': ground_truth,
            'inference': inference_texts,
            'audio_id': base_id,  # base name without extension
            'title': f'{prefix}_podcast',
            'original_full_path': file_path,
        }

        all_audio_info.append(audio_info)

    # Save one big JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_audio_info, f, indent=4, ensure_ascii=False)

    print(f" Saved inference JSON to {save_path}")



if __name__ == "__main__":
    podcast_csv_path = "/home/user/Documents/NEU_SFU/speech_error_classification/Whispering-LLaMA/data_preparation/tft_matched_output.csv"
    audio_dir = "/home/user/Documents/NEU_SFU/speech_error_classification/data/audio_data/tft_segmented_long_form_audio"
    save_path = "/home/user/Documents/NEU_SFU/speech_error_classification/Whispering-LLaMA/audio_features/tft_inferences.json"
    
    # Check if the path exists and is a file
    if os.path.exists(podcast_csv_path) and os.path.isfile(podcast_csv_path):
        print("The CSV file exists.")
    else:
        print("The CSV file does not exist.")
        
    data_df = pd.read_csv(podcast_csv_path)
    dataset_csv = load_dataset('csv', data_files=podcast_csv_path)

    df = data_df.copy()

    features_to_keep = ['podcastId', 'audioFile', 'recordTime','longFormError', 'longFormStart', 'longFormEnd', 'shortFormError', 'shortFormStart', 'shortFormEnd']

    df = df[features_to_keep]
    transcribe_audios(df, audio_dir, save_path)



end_time = time.time()

elapsed_time = end_time - start_time

elapsed_time_minutes = elapsed_time / 60

print(f"script runtime {elapsed_time_minutes:.2f}")

# Parse the metadata
#metadata_entries = parse_metadata_from_dataframe(df)


#print(metadata_entries)    


