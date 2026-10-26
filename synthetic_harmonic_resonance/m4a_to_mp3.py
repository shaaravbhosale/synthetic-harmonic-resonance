from pydub import AudioSegment
import os

def convert_m4a_to_mp3(input_path, output_path=None, bitrate="192k"):
    # Ensure ffmpeg is available
    if not AudioSegment.converter:
        raise EnvironmentError("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")

    # Load the .m4a file
    audio = AudioSegment.from_file(input_path, format="m4a")

    # Define output path
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".mp3"

    # Export as .mp3
    audio.export(output_path, format="mp3", bitrate=bitrate)
    print(f"Converted: {input_path} → {output_path}")

# Example usage
if __name__ == "__main__":
    input_file = "../inputs/Vaishnav-Jan-To-Shreya-Ghoshal.m4a"  # Replace with your file path
    convert_m4a_to_mp3(input_file)
