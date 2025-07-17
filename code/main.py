import os
import torch
import torchaudio
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "whisper-tiny")

# 确保模型路径存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

# 加载模型和处理器
try:
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("使用 GPU 进行推理")
    else:
        print("使用 CPU 进行推理")
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


def process_audio(audio_file_path):
    """
    使用torchaudio直接加载音频，忽略元数据，强制处理为标准格式
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    try:
        # 直接使用torchaudio加载，不依赖ffmpeg
        waveform, sample_rate = torchaudio.load(audio_file_path)

        # 重采样到16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)

        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 准备模型输入
        input_features = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        # 使用GPU（如果可用）
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        # 模型推理
        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        # 解码结果
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]

    except Exception as e:
        raise RuntimeError(f"音频处理失败: {str(e)}")


def predict(file):
    """
    Gradio接口函数，处理文件上传
    """
    if not file:
        return "请上传WAV音频文件"

    try:
        # 获取上传文件的临时路径
        audio_path = file.name
        return process_audio(audio_path)

    except Exception as e:
        return f"错误: {str(e)}"


def create_web_interface():
    """
    创建Gradio界面（完全兼容低版本Gradio）
    """
    test_data_dir = os.path.join(BASE_DIR, "..", "tests", "data")
    example_files = []

    # 检查是否有示例文件
    if os.path.exists(test_data_dir):
        wav_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith('.wav')]
        if wav_files:
            example_files = [os.path.join(test_data_dir, f) for f in wav_files[:2]]

    with gr.Blocks(title="Whisper 音频转录") as interface:
        gr.Markdown("## Whisper 音频转录系统")

        # 使用Markdown替代description参数提供详细说明
        gr.Markdown("""
        🚨 **重要提示**  
        - 仅支持 **WAV 格式** 文件（.wav）  
        - 系统会自动处理采样率（建议16kHz）和声道数  
        - 无需安装ffmpeg，所有处理均在Python中完成  
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # 仅使用必要参数，避免版本兼容性问题
                file_input = gr.File(
                    label="上传音频",
                    file_types=[".wav"]
                )
                submit_btn = gr.Button("开始转录")

                gr.Markdown("### 示例音频")
                if example_files:
                    # 直接传递完整路径，依赖Gradio自动处理
                    gr.Examples(
                        examples=example_files,
                        inputs=file_input,
                        cache_examples=False
                    )
                else:
                    gr.Markdown("没有找到可用的示例音频文件")

            with gr.Column(scale=2):
                text_output = gr.Textbox(label="转录结果", lines=10)

        submit_btn.click(
            fn=predict,
            inputs=file_input,
            outputs=text_output
        )

        # 底部添加额外说明
        gr.Markdown("""
        ### 技术细节  
        - 模型: Whisper Tiny  
        - 推理设备: {}  
        - 支持语言: 主要支持英语（可通过更换模型扩展）  
        """.format("GPU" if torch.cuda.is_available() else "CPU"))

    return interface


if __name__ == "__main__":
    interface = create_web_interface()
    interface.launch(share=False)