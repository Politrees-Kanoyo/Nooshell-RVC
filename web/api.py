# -*- coding: utf-8 -*-

import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rvc.infer.infer import HUBERT_BASE_PATH, OUTPUT_DIR, RVC_MODELS_DIR, convert_audio, get_vc, load_hubert, load_rvc_model
from rvc.infer.infer import rvc_edgetts_infer as _rvc_edgetts_infer
from rvc.infer.infer import rvc_infer as _rvc_infer
from rvc.infer.infer import text_to_speech
from rvc.modules.model_manager import download_from_url as _download_from_url
from rvc.modules.model_manager import upload_separate_files as _upload_separate_files
from rvc.modules.model_manager import upload_zip_file as _upload_zip_file
from web.gradio.components.modules import OUTPUT_FORMAT, edge_voices, get_folders
from web.gradio.install import MODELS as HUBERT_MODELS
from web.gradio.install import download_and_replace_model as _download_hubert_model


class MushroomRVCAPI:
    def __init__(self):
        os.makedirs(RVC_MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def voice_conversion(
        self,
        rvc_model: str,
        input_path: str,
        f0_method: str = "rmvpe",
        f0_min: int = 50,
        f0_max: int = 1100,
        rvc_pitch: int = 0,
        protect: float = 0.5,
        index_rate: float = 0.0,
        volume_envelope: float = 1.0,
        autopitch: bool = False,
        autopitch_threshold: float = 155.0,
        autotune: bool = False,
        autotune_strength: float = 1.0,
        output_format: str = "wav",
    ) -> str:
        try:

            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass

            result = _rvc_infer(
                rvc_model=rvc_model,
                input_path=input_path,
                f0_method=f0_method,
                f0_min=f0_min,
                f0_max=f0_max,
                rvc_pitch=rvc_pitch,
                protect=protect,
                index_rate=index_rate,
                volume_envelope=volume_envelope,
                autopitch=autopitch,
                autopitch_threshold=autopitch_threshold,
                autotune=autotune,
                autotune_strength=autotune_strength,
                output_format=output_format,
                progress=DummyProgress(),
            )

            return result

        except Exception as e:
            raise Exception(f"Ошибка при преобразовании голоса: {str(e)}")

    def text_to_speech_conversion(
        self,
        rvc_model: str,
        tts_text: str,
        tts_voice: str = "ru-RU-SvetlanaNeural",
        f0_method: str = "rmvpe",
        f0_min: int = 50,
        f0_max: int = 1100,
        rvc_pitch: int = 0,
        protect: float = 0.5,
        index_rate: float = 0.0,
        volume_envelope: float = 1.0,
        autopitch: bool = False,
        autopitch_threshold: float = 155.0,
        autotune: bool = False,
        autotune_strength: float = 1.0,
        output_format: str = "wav",
        tts_rate: int = 0,
        tts_volume: int = 0,
        tts_pitch: int = 0,
    ) -> Tuple[str, str]:
        try:

            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass

            synth_path, converted_path = _rvc_edgetts_infer(
                rvc_model=rvc_model,
                f0_method=f0_method,
                f0_min=f0_min,
                f0_max=f0_max,
                rvc_pitch=rvc_pitch,
                protect=protect,
                index_rate=index_rate,
                volume_envelope=volume_envelope,
                autopitch=autopitch,
                autopitch_threshold=autopitch_threshold,
                autotune=autotune,
                autotune_strength=autotune_strength,
                output_format=output_format,
                tts_voice=tts_voice,
                tts_text=tts_text,
                tts_rate=tts_rate,
                tts_volume=tts_volume,
                tts_pitch=tts_pitch,
                progress=DummyProgress(),
            )

            return synth_path, converted_path

        except Exception as e:
            raise Exception(f"Ошибка при TTS преобразовании: {str(e)}")

    def download_model_from_url(self, url: str, model_name: str) -> str:
        try:

            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass

            return _download_from_url(url, model_name, DummyProgress())
        except Exception as e:
            raise Exception(f"Ошибка при загрузке модели: {str(e)}")

    def upload_model_zip(self, zip_path: str, model_name: str) -> str:
        try:
            try:
                from gradio import Error as GradioError
            except ImportError:
                GradioError = Exception

            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass

            class FileWrapper:
                def __init__(self, path):
                    self.name = path

            result = _upload_zip_file(FileWrapper(zip_path), model_name, DummyProgress())

            return result

        except GradioError as e:
            error_msg = str(e)
            raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"Ошибка при загрузке ZIP модели: {str(e)}")

    def upload_model_files(self, pth_path: str, index_path: Optional[str], model_name: str) -> str:
        try:

            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass

            class FileWrapper:
                def __init__(self, path):
                    self.name = path

            pth_file = FileWrapper(pth_path)
            index_file = FileWrapper(index_path) if index_path else None

            return _upload_separate_files(pth_file, index_file, model_name, DummyProgress())
        except Exception as e:
            raise Exception(f"Ошибка при загрузке файлов модели: {str(e)}")

    def install_hubert_model(self, model_name: str = "hubert_base.pt", custom_url: Optional[str] = None) -> str:
        try:

            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass

            return _download_hubert_model(model_name, custom_url, DummyProgress())
        except Exception as e:
            raise Exception(f"Ошибка при установке HuBERT модели: {str(e)}")

    def get_available_models(self) -> List[str]:
        return get_folders()

    def get_available_voices(self) -> Dict[str, List[str]]:
        return edge_voices

    def get_output_formats(self) -> List[str]:
        return OUTPUT_FORMAT

    def get_hubert_models(self) -> List[str]:
        return HUBERT_MODELS

    def get_f0_methods(self) -> List[str]:
        return ["rmvpe+", "rmvpe", "fcpe", "crepe", "crepe-tiny"]

    def convert_audio_format(self, input_path: str, output_path: str, output_format: str) -> None:
        try:
            convert_audio(input_path, output_path, output_format)
        except Exception as e:
            raise Exception(f"Ошибка при конвертации аудио: {str(e)}")

    async def synthesize_speech(
        self, voice: str, text: str, rate: int = 0, volume: int = 0, pitch: int = 0, output_path: str = None
    ) -> str:
        try:
            if output_path is None:
                output_path = os.path.join(OUTPUT_DIR, "synthesized_speech.wav")

            await text_to_speech(voice, text, rate, volume, pitch, output_path)
            return output_path
        except Exception as e:
            raise Exception(f"Ошибка при синтезе речи: {str(e)}")


api = MushroomRVCAPI()


def voice_conversion(*args, **kwargs):
    return api.voice_conversion(*args, **kwargs)


def text_to_speech_conversion(*args, **kwargs):
    return api.text_to_speech_conversion(*args, **kwargs)


def download_model_from_url(*args, **kwargs):
    return api.download_model_from_url(*args, **kwargs)


def upload_model_zip(*args, **kwargs):
    return api.upload_model_zip(*args, **kwargs)


def upload_model_files(*args, **kwargs):
    return api.upload_model_files(*args, **kwargs)


def install_hubert_model(*args, **kwargs):
    return api.install_hubert_model(*args, **kwargs)


def get_available_models():
    return api.get_available_models()


def get_available_voices():
    return api.get_available_voices()


def get_output_formats():
    return api.get_output_formats()


def convert_audio_format(*args, **kwargs):
    return api.convert_audio_format(*args, **kwargs)


async def synthesize_speech(*args, **kwargs):
    return await api.synthesize_speech(*args, **kwargs)


if __name__ == "__main__":
    print("Mushroom RVC API инициализирован")
    print(f"Доступные модели: {get_available_models()}")
