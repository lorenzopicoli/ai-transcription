import json
import subprocess
from pathlib import Path
# instantiate the pipeline
from pyannote.audio import Pipeline
import torchaudio
import datetime
from pyannote.audio.pipelines.utils.hook import ProgressHook
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.0",
  use_auth_token="xxx")


def parse_rttm_to_dict(rttm_file):
    speaker_turns = {}
    current_speaker = None
    current_start = None
    current_end = None

    with open(rttm_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # Skip invalid lines

            speaker_id = parts[7]
            start = int(parts[3].replace('.', ''))
            duration = int(parts[4].replace('.', ''))
            if duration < 400:
              continue
            end = start + duration

            # If the same speaker is speaking consecutively, extend the current segment
            if speaker_id == current_speaker and current_start is not None:
                last_speaker = speaker_turns[(current_start, current_end)]
                if last_speaker == speaker_id:
                    speaker_turns.pop((current_start, current_end))
                    speaker_turns[(current_start, end)] = speaker_id
                    current_end = end
            else:
                current_speaker = speaker_id
                current_start = start
                current_end = end
                speaker_turns[(start, end)] = speaker_id
    
    new = {}
    last_key = None
    last_value = None
    for (start, end), speaker in speaker_turns.items():
      new[(start, end)] = speaker
      if last_key is None:
        last_key = (start, end)
        last_value = speaker
        continue
      new.pop(last_key)
      new[last_key[0], start - 1] = last_value
      
      last_key = (start, end)
      last_value = speaker
    print(new)

    return new
  
def is_within_tolerance_range(number, range_start, range_end, tolerance):
    # Expanding the range by the tolerance
    lower_bound = range_start - tolerance
    upper_bound = range_end + tolerance

    # Check if the number falls within the expanded range
    return lower_bound <= number <= upper_bound


def find_speaker_for_time(speaker_turns, start_time, end_time):
    # print(speaker_turns, time)

    print('-------------------------------', start_time, end_time)
    start_match = None
    end_match = None
    for (start, end), speaker in speaker_turns.items():
      match_start = False
      match_end = False
      if is_within_tolerance_range(start_time, start, end, 10):
        start_match = (start, end)
        match_start = True
        
      if is_within_tolerance_range(end_time, start, end, 10):
        end_match = (start, end)
        match_end = True
        
      if match_end and match_start:
        print(f'Matched fully {start}, {end}, {speaker}')
        return speaker
      
      if start_match is not None and end_match is not None:
        break
    
    if start_match is None or end_match is None:
      return "Unknown"
    print(f'start_match {start_match}, end_match {end_match}')
    # return speaker_turns[start_match]

    start_similarity = abs(start_match[1] - start_time)
    end_similarity = (end_time - end_match[0])
    if start_similarity >= end_similarity:
      return speaker_turns[start_match]
    else:
      return speaker_turns[end_match]


def match_transcript_with_speakers(transcription_json, speaker_turns):
    output_md = transcription_json.replace('.json', '.md')
    with open(transcription_json, 'r') as file, open(output_md, 'w') as outfile:
        data = json.load(file)
        last_speaker = ""
        for item in data['transcription']:
            start_time = int(item['offsets']['from'])
            end_time = int(item['offsets']['to'])
            chosen_speaker = find_speaker_for_time(speaker_turns, start_time, end_time)
            text = item['text'].strip()
            print(f'Text {text}. Whisper time {start_time} - {end_time}. chosen_speaker {chosen_speaker}')
            if text == '':
              continue
            if chosen_speaker == last_speaker or chosen_speaker == '': 
                outfile.write(f'{text} ')
            else:
                if last_speaker != "":
                    outfile.write('\n')
                outfile.write(f'*{str(datetime.timedelta(seconds=start_time // 1000))}* **{chosen_speaker}:** ')
                outfile.write(f'{text} ')
                last_speaker = chosen_speaker

    # Read the contents of the file
    with open(output_md, 'r') as file:
        content = file.read()

    # Replace '[ Silence ]' with '*Silence*'
    content = content.replace('[ Silence ]', '*Silence*')

    # Write the modified contents back to the file
    with open(output_md, 'w') as file:
        file.write(content)
        
        
class ProgressHook:
    def __init__(self):
        # Initialize any necessary variables
        pass

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if completed is None:
            completed = total = 1

        if not hasattr(self, "step_name") or step_name != self.step_name:
            self.step_name = step_name
            print(f"Starting step: {self.step_name}")

        # Update progress display
        progress_percentage = (completed / total) * 100 if total > 0 else 0
        print(f"Progress of '{self.step_name}': {completed}/{total} ({progress_percentage:.2f}%)")

        # Print completion message when step is completed
        if completed >= total:
            print(f"Completed step: {self.step_name}\n")

    def __enter__(self):
        # This method is called when entering the context (the with statement)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # This method is called when exiting the context
        pass




base_folder = '/Users/lorenzopicoli/recordings'

# Specify the directory you want to scan
directory = Path(f'{base_folder}/recordings-in')

# Filter out directory names, leaving only files
files = [f for f in directory.glob('*') if f.is_file() and (f.suffix == '.mp3' or f.suffix == '.m4a')]

for file in files:
  print(file.stem)
  try:
    convert_command = [
      'ffmpeg', 
      '-i',
      f'{base_folder}/recordings-in/{file.name}',
      '-ar',
      '16000',
      '-ac',
      '2',
      '-y',
      f'{base_folder}/final-out/{file.stem}.wav',
    ]
    subprocess.run(convert_command, check=True)

        
    transcribe_command = [
      '/Users/lorenzopicoli/Downloads/whisper.cpp/main',
      '-m', '/Users/lorenzopicoli/Downloads/whisper.cpp/models/ggml-large-v2-q5_0.bin',
      f'{base_folder}/final-out/{file.stem}.wav',
      '-l', 'en',
      # '-bs', '5',
      '-et', '2.8',
      '-mc', '64',
      '-ml', '1',
      '-sow',
      '-oj',
      '-of', f'{base_folder}/transcription-out/{file.stem}'
    ]

    completed_process = subprocess.run(transcribe_command, check=True)
    
    with ProgressHook() as hook:
     # load to memory
     waveform, sample_rate = torchaudio.load(f'{base_folder}/final-out/{file.stem}.wav')
     diarization = pipeline({'waveform': waveform, 'sample_rate': sample_rate}, num_speakers=2, hook=hook)

    # dump the diarization output to disk using RTTM format
    with open(f'{base_folder}/transcription-out/{file.stem}.rttm', "w") as rttm:
     diarization.write_rttm(rttm)

    rttm_turns = parse_rttm_to_dict(f'{base_folder}/transcription-out/{file.stem}.rttm')
    match_transcript_with_speakers(f'{base_folder}/transcription-out/{file.stem}.json', rttm_turns)
  except subprocess.CalledProcessError as e:
    # Handle the error case
    print("An error occurred:", str(e))         

