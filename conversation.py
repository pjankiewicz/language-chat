# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import time
import warnings
from typing import List

from record_unlimited import record_unlimited

warnings.filterwarnings("ignore")


try:
    import ffmpeg
except ImportError:
    raise ImportError(
        "Failed to import ffmpeg, please install ffmpeg with `brew install ffmpeg & pip install"
        " ffmpeg`"
    )

try:
    import sounddevice as sd
except ImportError:
    raise ImportError(
        "Failed to import sounddevice, please install sounddevice with `pip install sounddevice`"
    )

try:
    import soundfile as sf
except:
    raise ImportError(
        "Failed to import soundfile, please install soundfile with `pip install soundfile`"
    )

try:
    import emoji
except ImportError:
    raise ImportError(
        "Failed to import emoji, please check the "
        "correct package at https://pypi.org/project/emoji/"
    )

try:
    import numpy
except ImportError:
    raise ImportError(
        "Failed to import numpy, please check the "
        "correct package at https://pypi.org/project/numpy/1.24.1/"
    )

try:
    import whisper
except ImportError:
    raise ImportError(
        "Failed to import whisper, please check the "
        "correct package at https://pypi.org/project/openai-whisper/"
    )

try:
    from xinference.client import Client
    from xinference.types import ChatCompletionMessage
except ImportError:
    raise ImportError(
        "Falied to import xinference, please check the "
        "correct package at https://pypi.org/project/xinference/"
    )

# ------------------------------------- global variable initialization ---------------------------------------------- #
logger = logging.getLogger(__name__)
# global variable to store the audio device choices.
audio_devices = "-1"

# ----------------------------------------- decorator libraries ----------------------------------------------------- #
emoji_man = "\U0001F9D4"
emoji_women = emoji.emojize(":woman:")
emoji_system = emoji.emojize(":robot:")
emoji_user = emoji.emojize(":supervillain:")
emoji_speaking = emoji.emojize(":speaking_head:")
emoji_sparkiles = emoji.emojize(":sparkles:")
emoji_jack_o_lantern = emoji.emojize(":jack-o-lantern:")
emoji_microphone = emoji.emojize(":studio_microphone:")
emoji_rocket = emoji.emojize(":rocket:")


# --------------------------------- supplemented util to get the record --------------------------------------------- #
def get_audio_devices() -> str:
    global audio_devices

    if audio_devices != "-1":
        return str(audio_devices)

    devices = sd.query_devices()
    print("\n")
    print(emoji_microphone, end="")
    print("  Audio devices:")
    print(devices)
    print(emoji_microphone, end="")
    # audio_devices = input("  Please select the audio device you want to record: ")
    audio_devices = "0"
    return audio_devices


q: queue.Queue = queue.Queue()


def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


# function to take audio input and transcript it into text-file.
def format_prompt(model, audio_input) -> str:
    # the second parameters of transcribe enable us to define the language we are speaking.
    return model.transcribe(audio_input)["text"]


# transcript the generated chatbot word to audio output so the user will hear the result.
def text_to_audio(response, voice_id):
    # for audio output, we apply the mac initiated "say" command to provide. For Windows users, if
    # you want audio output, you can try on pyttsx3 or gtts package to see their functionality!

    text = response
    if voice_id == "Bob":
        voice = "Daniel"
    elif voice_id == "Alice":
        voice = "Karen"
    # anything not belongs to alice or bob are said by system voice.
    else:
        voice = "Moira"
    # Execute the "say" command and wait the command to be completed.
    process = subprocess.Popen(["say", "-v", voice, text])
    process.wait()


def chat_with_bot(
    format_input, chat_history, alice_or_bob_state, system_prompt, model_ref
):
    completion = model_ref.chat(
        prompt=format_input,
        system_prompt=system_prompt,
        chat_history=chat_history,
        generate_config={"max_tokens": 1024},
    )

    if alice_or_bob_state == "Alice":
        print(emoji_women, end="")
        print(" Alice:", end="")
    else:
        print(emoji_man, end="")
        print(" Bob:", end="")

    chat_history: List["ChatCompletionMessage"] = []

    content = completion["choices"][0]["message"]["content"]
    print(content)

    chat_history.append(ChatCompletionMessage(role="assistant", content=content))

    return content


# ---------------------------------------- The program will run from below: ------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e", "--endpoint", type=str, help="Xinference endpoint, required", required=True
    )
    args = parser.parse_args()

    endpoint = args.endpoint

    client = Client(endpoint)

    model_b = "wizardlm-v1.0"
    print(
        f"{emoji_rocket} Launching model {model_b}. The initial download of the model may require a certain"
        f" amount of time."
    )
    model_b_uid = client.launch_model(
        model_name=model_b,
        model_format="ggmlv3",
        model_size_in_billions=7,
        quantization="q4_0",
        n_ctx=2048,
    )
    model_b_ref = client.get_model(model_b_uid)

    # ---------- program finally start! ------------ #
    chat_history = []
    alice_or_bob_state = "0"
    print("")
    print(emoji_jack_o_lantern, end="")
    print(" Welcome to the Xorbits inference chatroom ", end="")
    print(emoji_jack_o_lantern)
    print(emoji_sparkiles, end="")
    print(
        " Say something with 'exit', 'quit', 'bye', or 'see you' to leave the chatroom ",
        end="",
    )
    print(emoji_sparkiles)

    # receive the username.
    print("")
    print(emoji_system, end="")
    username = "Magda" #input(welcome_prompt)
    # calibrate()

    # define names for the chatbots and create welcome message for chat-room.
    system_prompt_alice = (
        "Your name is Alice you're a helpful AI assistant."
    )
    system_prompt_bob = system_prompt_alice

    # we can change the scale of the model here, the bigger the model, the higher the accuracy.
    transcribe_model = whisper.load_model("large")

    welcome_prompt2 = (
        f": Nice to meet you, {username}"
    )

    print("")
    print(emoji_system, end="")
    print(welcome_prompt2)
    text_to_audio(welcome_prompt2, "0")

    while True:
        audio_input = record_unlimited()
        transcribed_text = format_prompt(transcribe_model, audio_input)

        # set up the separation between each chat block.
        print("")
        print(emoji_user, end="")
        print(f" {username}:", end="")
        print(transcribed_text)

        completion = model_b_ref.chat(
            prompt=transcribed_text,
            system_prompt=system_prompt_alice,
            chat_history=chat_history,
            generate_config={"max_tokens": 1024},
        )

        print(emoji_women, end="")
        print(" Alice:", end="")
        content = completion["choices"][0]["message"]["content"]
        print(content)

        chat_history.append(ChatCompletionMessage(role="user", content=transcribed_text))
        chat_history.append(ChatCompletionMessage(role="assistant", content=content))
        text_to_audio(content, alice_or_bob_state)
