from pydub import AudioSegment
import os

def parse_segments(txt_file):
    """
    Parse the custom txt file into a list of (filename, start, end) tuples.
    """
    segments = []
    current_id, start, end = None, None, None

    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ID:"):
                current_id = line.replace("ID:", "").strip()
            elif line.startswith("longFormStart:"):
                start = float(line.replace("longFormStart:", "").strip())
            elif line.startswith("longFormEnd:"):
                end = float(line.replace("longFormEnd:", "").strip())
            elif line.startswith("------------------------------"):
                if current_id and start is not None and end is not None:
                    segments.append((current_id, start, end))
                # reset for next block
                current_id, start, end = None, None, None
                
    
    return segments


def segment_audio(txt_file, audio_dir, output_dir, sample_rate=16000):
    os.makedirs(output_dir, exist_ok=True)
    segments = parse_segments(txt_file)
    #print('segment id or path name', segments)
    
    # Track segment counts per audio file
    file_segment_counter = {}
    
    for filename, start, end in segments:
        audio_path = os.path.join(audio_dir, filename)
        if not os.path.isfile(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue

        # Load audio file
        audio = AudioSegment.from_file(audio_path)

        # Convert to ms for pydub
        start_ms = start * 1000
        end_ms = end * 1000

        # Segment
        seg = audio[start_ms:end_ms]
        
        # Resample to 16kHz
        seg = seg.set_frame_rate(sample_rate).set_channels(1)

        # Save with descriptive name
        base, ext = os.path.splitext(os.path.basename(filename))
        
        file_segment_counter[base] = file_segment_counter.get(base, 0) + 1
        seg_idx = file_segment_counter[base]

        # Output name: base_segN.mp3
        #out_name = f"{base}_seg{seg_idx}{ext}"
        out_name = f"{base}_{int(start*1000)}_{int(end*1000)}{ext}"
        
        #out_name = f"{base}_{int(start_ms)}_{int(end_ms)}{ext}"
        #out_name = f"{base}{ext}"
        out_path = os.path.join(output_dir, out_name)
        seg.export(out_path, format=ext[1:])
        print(f"Saved: {out_path}")


# Example usage
if __name__ == "__main__":
    txt_file = "/home/user/Documents/NEU_SFU/speech_error_classification/annotation_data/unorganized_short_form_error/14_tft_lexical_short_form_error_mismatch.txt"        # your metadata txt file
    audio_dir = "/home/user/Documents/NEU_SFU/speech_error_classification/data/audio_data/lexical_substitution/tft"             # folder where full audio files are stored
    output_dir = "/home/user/Documents/NEU_SFU/speech_error_classification/data/audio_data/lexical_substitution/testing_tft"  # folder to save segmented audio
    segment_audio(txt_file, audio_dir, output_dir, sample_rate=16000)

