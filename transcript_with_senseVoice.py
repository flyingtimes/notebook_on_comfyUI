from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import modelscope,sys,os
from dotenv import load_dotenv
from openai import OpenAI
# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
load_dotenv(".env")
api_key = os.getenv("DEEPSEEK_API_KEY")
print(api_key)
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
url =  sys.argv[1]
file_id = url.split("v=")[-1]
print(file_id)
model = AutoModel(
    model="models/iic/SenseVoiceSmall",  
    vad_model="models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
  #  punc_model="models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    device="cuda:0",
)

# en
res = model.generate(
    input=f"audio_files/{file_id}.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
#model = AutoModel(model="models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
text = format_transcript(text)
#res = model.generate(input=text)
with open(f"transcript/{file_id}.txt", "w") as f:
    f.write(text)
print(text)
