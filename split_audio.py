import wave
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def load_file(filename):
    wave_reader = wave.open(filename, "rb")

    return wave_reader

def create_float_list(wave_reader, seconds):
    max_size = int.from_bytes(b"\xFF\xFF", "little")
    
    byte_list = wave_reader.readframes(int(wave_reader.getframerate() * seconds * 4))
    four_wide = [(byte_list[i*4:i*4+2], byte_list[i*4+2:i*4+4]) for i in range(len(byte_list) // 4)]
    channel1_list = [int.from_bytes(four_wide[i][0], "little") / max_size for i in range(len(four_wide))]
    channel2_list = [int.from_bytes(four_wide[i][1], "little") / max_size for i in range(len(four_wide))]
    count_list = [i for i in range(len(four_wide))]

    return channel1_list, channel2_list, count_list

def plot(count_list, channel1_list, channel2_list):
    plt.plot(count_list, channel1_list)
    plt.plot(count_list, channel2_list)
    plt.show()

def split_audio(wave_reader, timelength, outputdir, name):
    framerate = wave_reader.getframerate()
    channels = wave_reader.getnchannels()
    sampwidth = wave_reader.getsampwidth()
    total_frames = wave_reader.getnframes()
    frames_per_sample = framerate * timelength

    total_sections = total_frames // (framerate * timelength)
    for section in tqdm(range(total_sections)):
        path = outputdir + name + "{}.wav".format(section)
        wave_writer = wave.open(path, "wb")

        frames = wave_reader.readframes(frames_per_sample)

        wave_writer.setnchannels(channels)
        wave_writer.setsampwidth(sampwidth)
        wave_writer.setframerate(framerate)
        wave_writer.writeframes(frames)

        wave_writer.close()

def is_valid_output_dir(outputdir):
    if os.path.isdir(outputdir):
        return outputdir
    else:
        os.mkdir(outputdir)
        return outputdir

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", help="Input wav file")
    parser.add_argument("outputdir", type=is_valid_output_dir, help="Output directory for wav audio sections")
    parser.add_argument("--timelength", type=int, default=10, help="Length of audio sections in seconds")
    parser.add_argument("--name", type=str, default="sample", help="Base name of the audio section output files")

    args = parser.parse_args()


    wave_reader = load_file(args.filename)
    split_audio(wave_reader, args.timelength, args.outputdir, args.name)


if __name__ == "__main__":
    main()
