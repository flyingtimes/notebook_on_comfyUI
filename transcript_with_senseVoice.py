from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import modelscope,sys,os,requests
from dotenv import load_dotenv
from openai import OpenAI
import shutil
from typing import Annotated, AsyncGenerator, Literal
from pydub import AudioSegment

import httpx
import ormsgpack
from pydantic import AfterValidator, BaseModel, conint

load_dotenv(".env")
api_key = os.getenv("DEEPSEEK_API_KEY")
fish_api_key = os.getenv("FISH_API_KEY")
import subprocess

def merge_audio_files(input_files, output_file):
    input_args = ""
    outfile = output_file.split(".")[0]+".mp3"
    for file in input_files:
        input_args += f"-i {file} "
    command = f"ffmpeg {input_args} -filter_complex concat=n={len(input_files)}:v=0:a=1 -f mp3 {outfile}"
    subprocess.run(command, shell=True)

class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "mp3"
    mp3_bitrate: Literal[64, 128, 192] = 128
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True

def write_wave(atext,path):
    filename=path.split(".")[0]
    ext = path.split(".")[-1]
    mytexts = atext.split("\n")
    i=0
    parts =  []
    for mytext in mytexts:
        print(mytext)
        i=i+1
        if (len(mytext)<2):
            continue
        parts.append(f"{filename}_{i}.{ext}")

        request = ServeTTSRequest(
            text=mytext,
            references=[
                ServeReferenceAudio(
                    #audio=open("dingzhen.mp3", "rb").read(),
                    #text="如果这个世界都充满欢乐，那该多好啊！",
                    #audio=open("xuejie.mp3", "rb").read(),
                    #text="我们致力于帮助以色列自卫，但同时，我们继续通过外交努力防止这场冲突的重大升级。",
                    audio=open("output.wav", "rb").read(),
                    text="冥想对于很多不熟悉它的人来说是一种近乎于玄学的神秘存在，但随着现代社会节奏加快和信息大爆炸，人们每天被巨大压力和焦虑裹挟",
                )
            ],
        )
        
        with (
            httpx.Client() as client,
            open(f"{filename}_{i}.{ext}", "wb") as f,
        ):
            with client.stream(
                "POST",
                "https://api.fish.audio/v1/tts",
                content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
                headers={
                    "api-key": fish_api_key,
                    "content-type": "application/msgpack",
                },
                timeout=None,
            ) as response:
                for chunk in response.iter_bytes():
                    f.write(chunk) 
    merge_audio_files(parts, path)
def second_generate(text):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/beta")
    prompt = '''
请根据用户给出的信息，运用以下技巧对内容进行二次创作,请用中文输出结果：
- **句型与词汇调整**：通过替换原文中的句子结构和词汇以传达同样的思想。
- **内容拓展与插入**：增添背景知识、实例和历史事件，以丰富文章内容，并降低关键词密度。
- **避免关键词使用**：避免使用原文中的明显关键词或用其它词汇替换。
- **结构与逻辑调整**：重新排列文章的结构和逻辑流程，确保与原文的相似度降低。
- **语言翻译**：请翻译成中文。
'''
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
    ],
        max_tokens=8192,
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content
def format_transcript(text):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/beta")
    prompt = f"请将下面用户给出的内容按照正常人说话的方式输出带标点符号，但不带表情符号的文本"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
    ],
        max_tokens=8192,
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content

#modelscope.snapshot_download("iic/SenseVoiceSmall",cache_dir="models/")
#modelscope.snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",cache_dir="models/")
#modelscope.snapshot_download("ct-punc",cache_dir="models/")

def gen_transcript(filename):
    path = os.path.join("audio_files", filename)
    destination_path = os.path.join("audio_files_done", filename)
    output_wav_path = os.path.join("read_output", filename)
    file_id = path.split("\\")[-1]
    print(f"processing {path}")
    model = AutoModel(
        model="models/iic/SenseVoiceSmall",  
        vad_model="models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 30000},
    #  punc_model="models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        device="cuda:0",
    )

    # en
    res = model.generate(
        input=path,
        cache={},
        language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    text = format_transcript(text).rstrip("\n")
    with open(f"transcript/{file_id}.txt", "w",encoding='utf-8') as f:
        f.write(text)
    text = second_generate(text)
    with open(f"second_modify/{file_id}.txt", "w",encoding='utf-8') as f:
        f.write(text)
    print("generating voice.")
    write_wave(text,output_wav_path)
    if not os.path.exists("audio_files_done"):
        os.makedirs("audio_files_done")
    shutil.move(path, destination_path)
    return text
def process_files_in_dir(directory):
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            gen_transcript(filename)

def wav_files_in_dir(path):
    with open(path, "r",encoding='utf-8') as f:
        text=f.read()
    write_wave(text,"read_output\\A-UV7Z13uAQ.m4a")

directory_path = 'audio_files'
process_files_in_dir(directory_path)
