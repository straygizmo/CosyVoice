from modelscope import snapshot_download
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import os

# https://github.com/FunAudioLLM/CosyVoice/blob/main/FAQ.md
os.environ['PYTHONPATH'] = 'third_party/Matcha-TTS'

snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='pretrained_models/CosyVoice-300M-25Hz')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')

#cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False)
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=False)


# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
prompt_speech_16k = load_wav('./asset/Somecallmenature.wav', 16000)
prompt_text = "Some call me nature, others call me mother nature. I've been here for over four point five billion years, twenty two thousand five hundred times longer than you."

# zero_shot usage
#z1 = cosyvoice.inference_zero_shot("I haven't seen anything I was uh the there's a story from what today's does two days ago", prompt_text, prompt_speech_16k, stream=False, text_frontend=False)
#for i, j in enumerate(z1):
#    torchaudio.save('zero_shot_{}_en.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

#z1 = cosyvoice.inference_zero_shot("<strong>I haven't seen anything</strong> I was uh the there's a story from what today's does <strong>two days</strong> ago", prompt_text, prompt_speech_16k, stream=False, text_frontend=False)
#for i, j in enumerate(z1):
#    torchaudio.save(f'zero_shot_{i}_strong.wav', j['tts_speech'], cosyvoice.sample_rate)

#z1 = cosyvoice.inference_zero_shot("[breath]I haven't seen anything[laughter] I was uh the there's a [noise]story from what today's does <strong>two days</strong> ago", prompt_text, prompt_speech_16k, stream=False, text_frontend=False)
#for i, j in enumerate(z1):
#    torchaudio.save(f'zero_shot_{i}_[breath][laughter][noise].wav', j['tts_speech'], cosyvoice.sample_rate)

z1 = cosyvoice.inference_zero_shot("<laughter>I haven't seen anything</laughter> I was uh the there's a [mn]story from what today's does [vocalized-noise]two days ago", prompt_text, prompt_speech_16k, stream=False, text_frontend=False)
for i, j in enumerate(z1):
    torchaudio.save(f'zero_shot_{i}_laughterTAG.wav', j['tts_speech'], cosyvoice.sample_rate)

#for i, j in enumerate(cosyvoice.inference_zero_shot("I haven't seen anything I was uh the there's a story from what today's does two days ago[laughter]", prompt_text, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save('zero_shot_{}_en_laughter.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
#for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
#for i, j in enumerate(cosyvoice.inference_instruct2()'收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
