#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from assets.model_installer import check_and_install_models

check_and_install_models()

import argparse
import os
import sys

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from web.api import (
    MushroomRVCAPI,
    convert_audio_format,
    download_model_from_url,
    get_available_models,
    get_available_voices,
    get_output_formats,
    install_hubert_model,
    synthesize_speech,
    text_to_speech_conversion,
    upload_model_files,
    upload_model_zip,
    voice_conversion,
)

CURRENT_LANGUAGE = "ru"
if "--lang" in sys.argv:
    lang_index = sys.argv.index("--lang")
    if lang_index + 1 < len(sys.argv):
        lang = sys.argv[lang_index + 1]
        if lang in ["ru", "en"]:
            CURRENT_LANGUAGE = lang

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.config["SECRET_KEY"] = "mushroom-rvc-web-ui"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

I18N = {
    "ru": {
        "title": "Mushroom RVC Web UI",
        "voice_conversion": "Преобразование голоса",
        "text_to_speech": "Синтез речи",
        "model_management": "Управление моделями",
        "select_model": "Выберите модель RVC",
        "select_audio_file": "Выберите аудиофайл",
        "upload_audio": "Загрузить аудиофайл",
        "pitch_shift": "Сдвиг высоты тона",
        "f0_method": "Метод F0",
        "consonant_protection": "Защита согласных",
        "index_influence": "Влияние индекса",
        "output_format": "Формат вывода",
        "convert": "Преобразовать",
        "enter_text": "Введите текст для синтеза",
        "select_voice": "Выберите голос TTS",
        "speech_speed": "Скорость речи",
        "volume": "Громкость",
        "rvc_pitch": "Высота тона RVC",
        "f0_min": "F0 мин",
        "f0_max": "F0 макс",
        "volume_envelope": "Огибающая громкости",
        "synthesize_convert": "Синтезировать и преобразовать",
        "download_model": "Загрузить модель по URL",
        "model_url": "URL модели",
        "model_name": "Имя модели",
        "download": "Скачать",
        "upload_zip": "Загрузить ZIP модель",
        "select_zip": "Выберите ZIP файл",
        "upload": "Загрузить",
        "hubert_management": "Управление HuBERT",
        "select_hubert_model": "Выберите HuBERT модель",
        "install": "Установить",
        "processing": "Обработка...",
        "success": "Успешно!",
        "error": "Ошибка",
        "download_result": "Скачать результат",
        "advanced_settings": "Дополнительные настройки",
        "auto_pitch": "Автопитч",
        "auto_tune": "Автотюн",
        "clean_audio": "Очистка аудио",
        "clean_strength": "Сила очистки",
        "upsampling": "Апсемплинг",
        "result": "Результат",
        "custom_url_optional": "Пользовательский URL (опционально)",
        "autopitch_threshold": "Порог автопитча",
        "autotune_strength": "Сила автотюна",
    },
    "en": {
        "title": "Mushroom RVC Web UI",
        "voice_conversion": "Inference",
        "text_to_speech": "Text to Speech",
        "model_management": "Model Management",
        "select_model": "Select RVC Model",
        "select_audio_file": "Select Audio File",
        "upload_audio": "Upload Audio File",
        "pitch_shift": "Pitch Shift",
        "f0_method": "F0 Method",
        "consonant_protection": "Consonant Protection",
        "index_influence": "Index Influence",
        "output_format": "Output Format",
        "convert": "Convert",
        "enter_text": "Enter text to synthesize",
        "select_voice": "Select TTS Voice",
        "speech_speed": "Speech Speed",
        "volume": "Volume",
        "rvc_pitch": "RVC Pitch",
        "f0_min": "F0 Min",
        "f0_max": "F0 Max",
        "volume_envelope": "Volume Envelope",
        "synthesize_convert": "Synthesize and Convert",
        "download_model": "Download Model by URL",
        "model_url": "Model URL",
        "model_name": "Model Name",
        "download": "Download",
        "upload_zip": "Upload ZIP Model",
        "select_zip": "Select ZIP File",
        "upload": "Upload",
        "hubert_management": "HuBERT Management",
        "select_hubert_model": "Select HuBERT Model",
        "install": "Install",
        "processing": "Processing...",
        "success": "Success!",
        "error": "Error",
        "download_result": "Download Result",
        "advanced_settings": "Advanced Settings",
        "auto_pitch": "Auto Pitch",
        "auto_tune": "Auto Tune",
        "clean_audio": "Clean Audio",
        "clean_strength": "Clean Strength",
        "upsampling": "Upsampling",
        "result": "Result",
        "custom_url_optional": "Custom URL (optional)",
        "autopitch_threshold": "Autopitch Threshold",
        "autotune_strength": "Autotune Strength",
    },
}

api = MushroomRVCAPI()

UPLOAD_FOLDER = "temp_uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "aiff", "ac3"}
ALLOWED_MODEL_EXTENSIONS = {"zip", "pth", "index"}


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@app.route("/")
def index():
    return render_template("index.html", i18n=I18N[CURRENT_LANGUAGE], lang=CURRENT_LANGUAGE)


@app.route("/api/i18n")
def get_i18n():
    return jsonify(I18N[CURRENT_LANGUAGE])


@app.route("/api/set-language", methods=["POST"])
def set_language():
    global CURRENT_LANGUAGE
    data = request.get_json()
    lang = data.get("language", "ru")

    if lang in I18N:
        CURRENT_LANGUAGE = lang
        return jsonify({"success": True, "language": lang})
    else:
        return jsonify({"success": False, "error": "Unsupported language"}), 400


@app.route("/api/models")
def get_models():
    try:
        models = get_available_models()
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/voices")
def get_voices():
    try:
        voices = get_available_voices()
        return jsonify({"success": True, "voices": voices})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/formats")
def get_formats():
    try:
        formats = get_output_formats()
        return jsonify({"success": True, "formats": formats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/f0-methods")
def get_f0_methods():
    try:
        methods = api.get_f0_methods()
        return jsonify({"success": True, "methods": methods})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/hubert-models")
def get_hubert_models():
    try:
        models = api.get_hubert_models()
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/voice-conversion", methods=["POST"])
def api_voice_conversion():
    try:
        if "audio_file" not in request.files:
            return jsonify({"success": False, "error": "Файл не найден"})

        file = request.files["audio_file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Файл не выбран"})

        if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({"success": False, "error": "Неподдерживаемый формат файла"})

        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        rvc_model = request.form.get("rvc_model")
        f0_method = request.form.get("f0_method", "rmvpe+")
        f0_min = int(request.form.get("f0_min", 50))
        f0_max = int(request.form.get("f0_max", 1100))
        rvc_pitch = int(request.form.get("rvc_pitch", 0))
        protect = float(request.form.get("protect", 0.5))
        index_rate = float(request.form.get("index_rate", 0.7))
        volume_envelope = float(request.form.get("volume_envelope", 1.0))
        autopitch = request.form.get("autopitch") == "true"
        autopitch_threshold = float(request.form.get("autopitch_threshold", 200.0))
        autotune = request.form.get("autotune") == "true"
        autotune_strength = float(request.form.get("autotune_strength", 0.8))
        output_format = request.form.get("output_format", "wav")

        output_path = voice_conversion(
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
        )

        os.remove(input_path)

        return jsonify({"success": True, "output_path": output_path, "download_url": f"/download/{os.path.basename(output_path)}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/tts-conversion", methods=["POST"])
def api_tts_conversion():
    try:
        data = request.get_json()

        rvc_model = data.get("rvc_model")
        tts_text = data.get("tts_text")
        tts_voice = data.get("tts_voice")
        tts_rate = int(data.get("tts_rate", 0))
        tts_volume = int(data.get("tts_volume", 0))
        tts_pitch = int(data.get("tts_pitch", 0))
        rvc_pitch = int(data.get("rvc_pitch", 0))
        protect = float(data.get("protect", 0.5))
        index_rate = float(data.get("index_rate", 0.7))
        volume_envelope = float(data.get("volume_envelope", 1.0))
        output_format = data.get("output_format", "wav")

        synth_path, converted_path = text_to_speech_conversion(
            rvc_model=rvc_model,
            tts_text=tts_text,
            tts_voice=tts_voice,
            tts_rate=tts_rate,
            tts_volume=tts_volume,
            tts_pitch=tts_pitch,
            rvc_pitch=rvc_pitch,
            protect=protect,
            index_rate=index_rate,
            volume_envelope=volume_envelope,
            output_format=output_format,
        )

        return jsonify(
            {
                "success": True,
                "synth_path": synth_path,
                "converted_path": converted_path,
                "download_url": f"/download/{os.path.basename(converted_path)}",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/download-model", methods=["POST"])
def api_download_model():
    try:
        data = request.get_json()
        url = data.get("url")
        model_name = data.get("model_name")

        result = download_model_from_url(url=url, model_name=model_name)
        return jsonify({"success": True, "message": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/upload-model-zip", methods=["POST"])
def api_upload_model_zip():
    try:
        if "model_file" not in request.files:
            return jsonify({"success": False, "error": "Файл не найден"})

        file = request.files["model_file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Файл не выбран"})

        if not file.filename.lower().endswith(".zip"):
            return jsonify({"success": False, "error": "Требуется ZIP файл"})

        model_name = request.form.get("model_name")
        if not model_name or model_name.strip() == "":
            return jsonify({"success": False, "error": "Имя модели не указано"})

        filename = secure_filename(file.filename)
        zip_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            file.save(zip_path)
            print(f"[DEBUG] Файл сохранен: {zip_path}, существует: {os.path.exists(zip_path)}")
        except Exception as save_error:
            print(f"[ERROR] Ошибка сохранения файла: {save_error}")
            return jsonify({"success": False, "error": f"Ошибка сохранения файла: {str(save_error)}"})
        try:
            print(f"[DEBUG] Начинаем загрузку модели из: {zip_path}")
            result = upload_model_zip(zip_path=zip_path, model_name=model_name)
            print(f"[DEBUG] Загрузка завершена успешно: {result}")

            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                    print(f"[DEBUG] Временный файл удален: {zip_path}")
                except Exception as remove_error:
                    print(f"[WARNING] Не удалось удалить временный файл: {remove_error}")

            return jsonify({"success": True, "message": result})
        except Exception as upload_error:
            print(f"[ERROR] Ошибка загрузки модели: {upload_error}")

            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                    print(f"[DEBUG] Временный файл удален после ошибки: {zip_path}")
                except Exception as remove_error:
                    print(f"[WARNING] Не удалось удалить временный файл после ошибки: {remove_error}")

            return jsonify({"success": False, "error": str(upload_error)})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/install-hubert", methods=["POST"])
def api_install_hubert():
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        custom_url = data.get("custom_url")

        result = install_hubert_model(model_name=model_name, custom_url=custom_url if custom_url else None)

        return jsonify({"success": True, "message": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/download/<filename>")
def download_file(filename):
    try:
        # Поиск файла в папке вывода
        output_dir = "output/RVC_output"
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "Файл не найден"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "Файл слишком большой"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Страница не найдена"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Внутренняя ошибка сервера"}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mushroom RVC Web UI")
    parser.add_argument("--lang", choices=["ru", "en"], default="ru", help="Язык интерфейса (ru/en) / Interface language (ru/en)")
    parser.add_argument("--port", type=int, default=5000, help="Порт для веб-сервера")
    parser.add_argument("--host", default="0.0.0.0", help="Хост для веб-сервера")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")

    args = parser.parse_args()

    if args.lang == "ru":
        print("🍄 Mushroom RVC Web UI запущен!")
        print(f"📱 Откройте браузер и перейдите по адресу: http://localhost:{args.port}")
        print("🎤 Готов к преобразованию голоса!")
        print(f"🌐 Язык интерфейса: Русский")
    else:
        print("🍄 Mushroom RVC Web UI started!")
        print(f"📱 Open your browser and go to: http://localhost:{args.port}")
        print("🎤 Ready for voice conversion!")
        print(f"🌐 Interface language: English")

    app.run(debug=args.debug, host=args.host, port=args.port)
