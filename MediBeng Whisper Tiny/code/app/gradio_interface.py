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
    创建Gradio界面（蓝色主题，强调按钮）
    """
    test_data_dir = os.path.join(BASE_DIR, "..", "tests", "data")
    example_files = []

    # 检查是否有示例文件
    if os.path.exists(test_data_dir):
        wav_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith('.wav')]
        if wav_files:
            example_files = [os.path.join(test_data_dir, f) for f in wav_files[:2]]

    # 蓝色主题CSS（增强按钮样式）
    css = """
    /* 全局样式重置 + 蓝色主题 */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    body {
        background: linear-gradient(135deg, #e6f7ff 0%, #b3d9ff 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
        line-height: 1.6;
    }
   .gradio-container {
        max-width: 1200px;
        margin: 40px auto;
        padding: 30px;
        background-color: #fff;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(52, 152, 219, 0.15);
        border: 2px solid #3498db;
    }
    /* 标题样式 - 蓝色 */
    h1 {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        color: #2980b9;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(52, 152, 219, 0.3);
    }
    h2 {
        color: #3498db;
        font-size: 28px;
        text-align: center;
        margin-bottom: 15px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        display: inline-block;
    }
    h3 {
        color: #2980b9;
        font-size: 22px;
        margin-top: 30px;
        margin-bottom: 12px;
    }
    /* 文件上传区块 - 蓝色边框 */
   .block-file {
        background-color: #f5faff;
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
   .block-file:hover {
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
        transform: translateY(-2px);
        border-color: #2980b9;
    }
    /* 按钮样式 - 深蓝色强调 */
   .block-button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 16px 32px;
        font-size: 20px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 10px;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        width: 100%;
    }
   .block-button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #2c3e50 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(52, 152, 219, 0.6);
    }
   .block-button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.4);
    }
    /* 文本框样式 - 蓝色边框 */
   .block-textbox {
        background-color: #f5faff;
        border: 2px solid #3498db;
        border-radius: 8px;
        padding: 20px;
        min-height: 180px;
        font-size: 16px;
        color: #333;
        transition: all 0.3s ease;
    }
   .block-textbox:focus {
        border-color: #2980b9;
        box-shadow: 0 0 8px rgba(52, 152, 219, 0.3);
        outline: none;
    }
    /* 示例区块 - 浅蓝色背景 */
   .gr-box {
        margin-top: 20px;
        padding: 20px;
        background-color: #f5faff;
        border: 2px solid #3498db;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
   .gr-box:hover {
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.15);
        border-color: #2980b9;
    }
    /* 示例按钮样式 - 蓝色 */
   .gr-button {
        background-color: #3498db;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        margin-right: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.1);
    }
   .gr-button:hover {
        background-color: #2980b9;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.2);
    }
    /* 蓝色进度条 */
   .gradio-container progress {
        accent-color: #3498db;
    }
    """

    with gr.Blocks(title="Whisper 音频转录", css=css) as interface:
        # 主标题
        gr.Markdown("# Whisper 音频转录系统")

        with gr.Row():
            with gr.Column(scale=1):
                # 文件上传组件
                file_input = gr.File(
                    label="上传音频文件",
                    file_types=[".wav"],
                    elem_classes="block-file"
                )
                # 转录按钮（蓝色主题，强调样式）
                submit_btn = gr.Button("开始转录", elem_classes="block-button")

                # 示例音频标题 + 示例
                gr.Markdown("### 示例音频")
                if example_files:
                    gr.Examples(
                        examples=example_files,
                        inputs=file_input,
                        cache_examples=False
                    )
                else:
                    gr.Markdown("没有找到可用的示例音频文件")

            with gr.Column(scale=2):
                # 结果输出框
                text_output = gr.Textbox(
                    label="转录结果",
                    lines=10,
                    elem_classes="block-textbox"
                )

        # 绑定点击事件
        submit_btn.click(
            fn=predict,
            inputs=file_input,
            outputs=text_output
        )

    return interface


if __name__ == "__main__":
    # 禁用Gradio分析（避免网络请求超时）
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    interface = create_web_interface()
    interface.launch(share=False, debug=True)