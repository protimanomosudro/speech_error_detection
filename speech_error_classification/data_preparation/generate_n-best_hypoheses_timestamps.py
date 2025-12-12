from faster_whisper import WhisperModel
import json
import torch
from typing import List, Dict
import os 

def transcribe_with_timestamps(audio_path: str, model_size: str = "large", 
                              num_hypotheses: int = 5, word_timestamps: bool = True):
    """
    Transcribe with timestamps using faster-whisper
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    
    model = WhisperModel(
        model_size,
        device="cuda" if device == "cuda" else "cpu",
        compute_type=compute_type
    )
    
    hypotheses = []
    beam_sizes = [1, 2, 3, 4, 5,6,7,8,9,10][:num_hypotheses]
    # Initialize an empty list to collect all inference texts
    #inference_texts = []
    for i, beam_size in enumerate(beam_sizes):
        try:
            print(f"Generating hypothesis {i+1} with beam size {beam_size}")
            
            
            segments, info = model.transcribe(
                audio_path,
                beam_size=beam_size,
                language="en",
                vad_filter=True,
                temperature=0.0,
                best_of=10,
                word_timestamps=word_timestamps  # Enable word-level timestamps
            )
            
            # Collect all segments with timestamps
            segments_list = []
            for segment in segments:
                segment_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'words': []
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        segment_data['words'].append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        })
                
                segments_list.append(segment_data)
            
            # Combine text for full transcription
            full_text = " ".join([seg['text'] for seg in segments_list])
            
            hypotheses.append({
                'hypothesis_id': i + 1,
                'beam_size': beam_size,
                'full_text': full_text,
                'language': info.language,
                'language_probability': info.language_probability,
                'segments': segments_list,
                'num_segments': len(segments_list)
            })
            
            
        except Exception as e:
            print(f"Error with beam size {beam_size}: {e}")
            continue
    
    return hypotheses



def save_hypotheses(hypotheses, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hypotheses, f, indent=2, ensure_ascii=False)
        
        
def process_directory(input_dir: str, output_dir: str, model_size: str = "large", num_hypotheses: int = 5):
    # List supported audio extensions
    audio_exts = (".mp3", ".wav", ".flac", ".m4a")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_exts):
                audio_path = os.path.join(root, file)

                # compute relative path from input_dir
                rel_path = os.path.relpath(audio_path, input_dir)
                # replace extension with .json
                rel_json = os.path.splitext(rel_path)[0] + ".json"
                output_path = os.path.join(output_dir, rel_json)

                print(f"\nProcessing: {audio_path}\nSaving to: {output_path}")
                
                hypotheses = transcribe_with_timestamps(audio_path, model_size=model_size, num_hypotheses=num_hypotheses,word_timestamps=True)
                save_hypotheses(hypotheses, output_path)




if __name__ == "__main__":
    # Path for the input and output directory
    input_dir = "/home/user/Documents/NEU_SFU/speech_error_classification/data/audio_data/ac_segmented_log_form_audio"
    output_dir = "/home/user/Documents/NEU_SFU/speech_error_classification/transcript_outputs/"
    
    process_directory(input_dir, output_dir, model_size="large", num_hypotheses=5)
