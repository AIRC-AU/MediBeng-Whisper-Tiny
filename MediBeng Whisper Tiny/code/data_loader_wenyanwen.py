from datasets import load_dataset, Audio
from transformers import WhisperTokenizer, WhisperProcessor
import os
import logging
from gtts import gTTS
import hashlib

# 配置
MODEL_NAME = "openai/whisper-tiny"
LANGUAGE = "zh"
TASK = "translate"
SAMPLING_RATE = 16000
AUDIO_DIR = "D:/pycharmcode/jiqifanyi end/MediBeng-Whisper-Tiny-main/data/wenyan_audio"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def generate_filename(text):
    hash_val = hashlib.sha256(text.encode()).hexdigest()
    return os.path.join(AUDIO_DIR, f"{hash_val}.mp3")

def text_to_speech(text, path):
    if not os.path.exists(path):
        try:
            tts = gTTS(text, lang="zh-CN")
            tts.save(path)
        except Exception as e:
            logging.warning(f"gTTS error on text: {text[:30]}... — {e}")
    return path

def load_and_prepare_datasets():
    ensure_directory(AUDIO_DIR)
    logging.info("Loading WenYanWen dataset...")
    dataset = load_dataset("KaifengGGG/WenYanWen_English_Parallel", split="train[:300]")

    print("🚨 样本结构如下：")
    print(dataset[0])  # 打印第一个样本的字段，确认字段名如 'wenyanwen'、'english'

    def add_audio_path(example):
        text = example["wenyanwen"]  # ✅ 使用正确字段
        audio_path = generate_filename(text)
        example["audio"] = text_to_speech(text, audio_path)
        example["translation"] = example["english"]  # ✅ 翻译字段
        return example

    dataset = dataset.map(add_audio_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

    def prepare_example(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["translation"]).input_ids
        return batch

    dataset = dataset.map(prepare_example)
    split = dataset.train_test_split(test_size=0.1)

    return split["train"], split["test"], processor, tokenizer

# ✅ 主函数调用（不加这个脚本不会执行）
if __name__ == "__main__":
    load_and_prepare_datasets()
