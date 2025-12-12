import os
import json
import torch
import librosa
import numpy as np
from faster_whisper import WhisperModel
from tqdm import tqdm
from pathlib import Path
import re


def parse_metadata(txt_file):
    entries = []
    current_id, srt_time, end_time, ground_truth = None, None, None, None
    
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ID:"):
                
                raw = line.replace("ID:", "").strip()
                base = os.path.splitext(os.path.basename(raw))[0]
                current_id = base
                
            elif line.startswith("longFormStart:"):
                srt_time = line.split("longFormStart:")[1].strip()
                
            elif line.startswith("longFormEnd:"):
                #entry["longFormEnd"] = line.split("longFormEnd:")[1].strip()
                end_time = line.split("longFormEnd:")[1].strip()
            elif line.startswith("longFormError:"):
                ground_truth = (
                    line.replace("longFormError:", "").strip()
                    .replace(' <COMMA>', ',')
                    .replace(' <PERIOD>', '.')
                    .replace(' <QUESTIONMARK>', '?')
                    .replace(' <EXCLAMATIONPOINT>', '!')
                    .lower()
                )
            elif line.startswith("------------------------------"):
                if current_id and srt_time and end_time and ground_truth:
                    entries.append((current_id, srt_time, end_time, ground_truth))
                current_id,srt_time, end_time, ground_truth = None, None, None, None    
                
    return entries


def transcribe_audios(txt_file, audio_dir, save_path):
    # Setup Whisper model
    model_size = "small"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Parse metadata file
    entries = parse_metadata(txt_file)

    all_audio_info = []
    for idx, (base_id, srt_time, end_time, ground_truth) in enumerate(tqdm(entries, desc="Processing audios")):
        start_ms = int(float(srt_time) * 1000)
        end_ms = int(float(end_time) * 1000)
        seg_filename = f"{base_id}_{start_ms}_{end_ms}.mp3" ## refactor code to consider any type of audio files
        file_path = os.path.join(audio_dir, seg_filename)

        if not os.path.isfile(file_path):
            print(f" Audio file not found for segment: {seg_filename}")
            continue
        
        # Load audio file (preserve native sample rate)
        audio_array, sampling_rate = librosa.load(file_path, sr=None)
        audio_array_list = audio_array.astype(np.single).tolist()
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
        audio_info = {
            'segment_id': str(idx + 1),   # running count
            'ground_truth': ground_truth,
            'inference': inference_texts,
            'audio_id': base_id,  # base name without extension
            'title': 'Astronomy podcast',
            'original_full_path': file_path,
        }

        all_audio_info.append(audio_info)

    # Save one big JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_audio_info, f, indent=4, ensure_ascii=False)

    print(f" Saved inference JSON to {save_path}")


if __name__ == "__main__":
    txt_file = "/home/user/Documents/NEU_SFU/speech_error_classification/annotation_data/ordered_ac_lexical_short_form_error_mismatch.txt"
    audio_dir = "/home/user/Documents/NEU_SFU/speech_error_classification/data/audio_data/ac_segmented_log_form_audio"
    save_path = "/home/user/Documents/NEU_SFU/speech_error_classification/Whispering-LLaMA/audio_features/ac_inferences.json"

    transcribe_audios(txt_file, audio_dir, save_path)
