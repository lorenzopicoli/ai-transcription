import json
import subprocess
from pathlib import Path
# instantiate the pipeline
from pyannote.audio import Pipeline
import torchaudio
from pydub import AudioSegment
import argparse


import os

from pyannote.audio.pipelines.utils.hook import ProgressHook

temp_dir = Path(__file__).parent.joinpath('./temp')

parser = argparse.ArgumentParser(description="Transcribe and diarize anything",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-w", "--whisper", help="Path to whisper", required=True)
parser.add_argument(
    "-m", "--model", help="Path to whisper model", required=True)
parser.add_argument(
    "-i", "--in-dir", help="Input directory with audio files", required=True)
parser.add_argument("-o", "--out-dir",
                    help="Output directory for results", required=True)
parser.add_argument("-s", "--speakers",
                    help="Number of speakers in clips", required=True)

args = parser.parse_args()
config = vars(args)


class DiarizationProgressHook:
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
        print(
            f"Progress of '{self.step_name}': {completed}/{total} ({progress_percentage:.2f}%)")

        # Print completion message when step is completed
        if completed >= total:
            print(f"Completed step: {self.step_name}\n")

    def __enter__(self):
        # This method is called when entering the context (the with statement)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # This method is called when exiting the context
        pass


def parse_rttm(rttm_file):
    speaker_turns = {}
    current_speaker = None
    current_start = None
    current_end = None

    with open(rttm_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

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

    return speaker_turns


def format_time(ms):
    """Convert milliseconds to HH:MM:SS format."""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def extract_audio_track(input_file, output_file, start_time, end_time):
    audio = AudioSegment.from_wav(input_file)
    track = audio[start_time:end_time]
    track.export(output_file, format="wav")


def convert_to_wav(file_path, out_path):
    convert_command = [
        'ffmpeg',
        '-i',
        file_path,
        '-ar',
        '16000',
        '-ac',
        '2',
        '-y',
        out_path,
    ]
    subprocess.run(convert_command, check=True)


def diarize(file):
    with DiarizationProgressHook() as hook:
        # load to memory
        waveform, sample_rate = torchaudio.load(file)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token="xxxx")
        diarization = pipeline({'waveform': waveform, 'sample_rate': sample_rate}, num_speakers=int(
            config['speakers']), hook=hook)

        rttm_path = temp_dir.joinpath(file.with_suffix('.rttm'))

        with open(rttm_path, "w") as rttm:
            diarization.write_rttm(rttm)

    parsed = parse_rttm(rttm_path)

    return parsed


def whisper(in_path):
    out_path = temp_dir.joinpath('transcription-extract')
    transcribe_command = [
        config['whisper'],
        '-m', config['model'],
        in_path,
        '-l', 'en',
        # '-bs', '5',
        # '-et', '2.8',
        # '-mc', '64',
        # '-ml', '1',
        # '-sow',
        '-otxt',
        '-of', out_path
    ]
    subprocess.run(transcribe_command, check=True)
    with open(out_path.with_suffix('.txt'), 'r') as file:
        content = file.read()

    return content.replace('\n', '')


def create_temp():
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)


def remove_temp():
    os.rmdir(temp_dir)


def transcribe(in_dir, out_dir):
    create_temp()
    directory = Path(in_dir)
    out_directory = Path(out_dir)
    # Find all mp3 or m4a files in the inDir
    files = [f for f in directory.glob(
        '*') if f.is_file() and (f.suffix == '.mp3' or f.suffix == '.m4a')]

    for file in files:
        wav = out_directory.joinpath(file.stem + '.wav')
        convert_to_wav(file, wav)
        speaker_turns = diarize(wav)
        md_content = ''
        for (start, end), speaker in speaker_turns.items():
            extract_path = temp_dir.joinpath(
                f'extract-{file.stem}-{start}-{end}.wav')
            extract_audio_track(wav, extract_path, start, end)
            extract_transcription = whisper(extract_path)
            start_formatted = format_time(int(start))
            end_formatted = format_time(int(end))
            md_content = '\n\n'.join(
                [md_content, f"*{start_formatted}* - *{end_formatted}* **[[{speaker}]]**:{extract_transcription}"])
        with open(out_directory.joinpath(file.stem + '.md'), 'w') as md_file:
            md_file.write(md_content)
    remove_temp()


transcribe(config['in_dir'], config['out_dir'])
