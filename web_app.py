#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from assets.model_installer import check_and_install_models

check_and_install_models()

import argparse
import gc
import os
import subprocess
import sys
import threading
import time
import urllib.request

from flask import Flask, jsonify, render_template, request, send_file, stream_template
from flask_cloudflared import run_with_cloudflared
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from web.api import current_conversion_progress  # Глобальная переменная для отслеживания прогресса
from web.api import validate_file_exists  # Безопасная проверка существования файлов
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
if "--cloudflared" in sys.argv:
    run_with_cloudflared(app)

app.config["SECRET_KEY"] = "mushroom-rvc-web-ui"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

I18N = {
    "ru": {
        "title": "Mushroom RVC WebUI",
        "voice_conversion": "Изменить голос",
        "text_to_speech": "Текст в речь",
        "model_management": "Управление моделями",
        "select_model": "Выберите модель RVC",
        "select_audio_file": "Выберите аудиофайл",
        "upload_audio": "Загрузить аудиофайл",
        "pitch_shift": "Питч",
        "f0_method": "F0 метод",
        "consonant_protection": "Защита согласных",
        "index_influence": "Влияние индекса",
        "output_format": "Формат вывода",
        "convert": "Конвертировать",
        "enter_text": "Введите текст",
        "select_voice": "Выберите голос",
        "speech_speed": "Скорость",
        "volume": "Громкость",
        "rvc_pitch": "Питч RVC",
        "f0_min": "F0 мин",
        "f0_max": "F0 макс",
        "volume_envelope": "Огибающая громкости",
        "synthesize_convert": "Конвертировать",
        "download_model": "Скачать модель по URL",
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
        "title": "Mushroom RVC WebUI",
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


def validate_file_size(file, max_size_mb=500):
    """Валидация размера файла"""
    max_size_bytes = max_size_mb * 1024 * 1024

    # Получаем размер файла
    file.seek(0, 2)  # Переходим в конец файла
    file_size = file.tell()
    file.seek(0)  # Возвращаемся в начало

    if file_size > max_size_bytes:
        return False, f"Файл слишком большой. Максимальный размер: {max_size_mb}MB"

    if file_size == 0:
        return False, "Файл пустой"

    return True, None


def save_uploaded_file(file, upload_folder, allowed_extensions):
    """Безопасное сохранение загруженного файла с валидацией"""
    try:
        # Проверка наличия файла
        if not file or file.filename == "":
            return None, "Файл не выбран"

        # Проверка расширения файла
        if not allowed_file(file.filename, allowed_extensions):
            return None, f'Неподдерживаемый формат файла. Разрешены: {", ".join(allowed_extensions)}'

        # Проверка размера файла
        is_valid, error_msg = validate_file_size(file)
        if not is_valid:
            return None, error_msg

        # Создание безопасного имени файла
        filename = secure_filename(file.filename)
        if not filename:
            return None, "Недопустимое имя файла"

        # Создание уникального имени файла для избежания конфликтов
        base_name, ext = os.path.splitext(filename)
        unique_filename = f"{base_name}_{os.urandom(8).hex()}{ext}"

        # Сохранение файла
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)

        return file_path, None

    except Exception as e:
        return None, f"Ошибка сохранения файла: {str(e)}"


def cleanup_temp_file(file_path):
    """Безопасное удаление временного файла"""
    try:
        if file_path and validate_file_exists(file_path):
            os.remove(file_path)
            print(f"[DEBUG] Временный файл удален: {file_path}")
    except Exception as e:
        print(f"[WARNING] Не удалось удалить временный файл {file_path}: {e}")


def setup_localtunnel(port=5000):
    """Настройка и запуск локального туннеля через localtunnel"""
    try:
        print("Установка localtunnel...")
        # Проверяем, установлен ли npm
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Ошибка: npm не найден. Установите Node.js для использования localtunnel.")
            return None

        # Устанавливаем localtunnel глобально
        print("Устанавливаем localtunnel...")
        install_result = subprocess.run(["npm", "install", "-g", "localtunnel"], capture_output=True, text=True)
        if install_result.returncode != 0:
            print(f"❌ Ошибка установки localtunnel: {install_result.stderr}")
            return None

        print("Запуск localtunnel...")

        # Создаем файл для записи URL
        url_file = "url.txt"
        with open(url_file, "w") as file:
            file.write("")

        # Запускаем localtunnel в фоновом режиме
        def run_tunnel():
            try:
                with open(url_file, "w") as file:
                    process = subprocess.Popen(["lt", "--port", str(port)], stdout=file, stderr=subprocess.STDOUT, text=True)
                    # Ждем завершения процесса
                    process.wait()
            except Exception as e:
                print(f"Ошибка запуска туннеля: {e}")

        # Запускаем туннель в отдельном потоке
        tunnel_thread = threading.Thread(target=run_tunnel, daemon=True)
        tunnel_thread.start()

        # Ждем немного для инициализации туннеля
        time.sleep(5)

        try:
            # Получаем внешний IP
            endpoint_ip = urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode("utf8").strip()

            # Читаем URL туннеля
            tunnel_url = ""
            if os.path.exists(url_file):
                with open(url_file, "r") as file:
                    content = file.read()
                    # Ищем URL в выводе localtunnel
                    for line in content.split("\n"):
                        if "your url is:" in line:
                            tunnel_url = line.replace("your url is:", "").strip()
                            break
                        elif "https://" in line and "loca.lt" in line:
                            tunnel_url = line.strip()
                            break

            if tunnel_url:
                print(f"\n🌐 Локальный туннель запущен!")
                print(f"🔗 Share Link: \033[93m{tunnel_url}\033[0m")
                print(f"🔑 Password IP: {endpoint_ip}")
                print(f"💡 Используйте IP адрес как пароль при первом подключении\n")
                return tunnel_url
            else:
                print("❌ Не удалось получить URL туннеля")
                return None

        except Exception as e:
            print(f"❌ Ошибка получения информации о туннеле: {e}")
            return None

    except Exception as e:
        print(f"❌ Ошибка настройки localtunnel: {e}")
        return None


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


@app.route("/api/upload-audio", methods=["POST"])
def api_upload_audio():
    """Endpoint для загрузки аудиофайлов"""
    try:
        if "audio_file" not in request.files:
            return jsonify({"success": False, "error": "Файл не найден"})

        file = request.files["audio_file"]

        file_path, error_msg = save_uploaded_file(file, UPLOAD_FOLDER, ALLOWED_AUDIO_EXTENSIONS)
        if error_msg:
            return jsonify({"success": False, "error": error_msg})

        return jsonify({"success": True, "file_path": file_path, "filename": os.path.basename(file_path)})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    except RequestEntityTooLarge:
        return jsonify({"success": False, "error": "Файл слишком большой (максимум 500MB)"})


@app.route("/api/remove-audio", methods=["POST"])
def api_remove_audio():
    """Endpoint для удаления загруженных аудиофайлов"""
    try:
        data = request.get_json()
        file_path = data.get("file_path")

        if not file_path:
            return jsonify({"success": False, "error": "Путь к файлу не указан"})

        if validate_file_exists(file_path) and os.path.abspath(file_path).startswith(os.path.abspath(UPLOAD_FOLDER)):
            os.remove(file_path)
            return jsonify({"success": True, "message": "Файл успешно удален"})
        else:
            return jsonify({"success": False, "error": "Файл не найден или недоступен для удаления"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/conversion-progress")
def get_conversion_progress():
    """Endpoint для получения текущего прогресса конвертации"""
    try:
        from web.api import current_conversion_progress

        return jsonify({"success": True, "progress": current_conversion_progress})
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "progress": {
                    "progress": 0.0,
                    "current_step": 0,
                    "total_steps": 8,
                    "step_name": "Ошибка",
                    "description": "Ошибка получения прогресса",
                },
            }
        )


@app.route("/api/voice-conversion", methods=["POST"])
def api_voice_conversion():
    input_path = None
    try:
        if "audio_file_path" in request.form:
            input_path = request.form.get("audio_file_path")
            if not validate_file_exists(input_path):
                return jsonify({"success": False, "error": "Загруженный файл не найден"})
        elif "audio_file" in request.files:
            file = request.files["audio_file"]
            input_path, error_msg = save_uploaded_file(file, UPLOAD_FOLDER, ALLOWED_AUDIO_EXTENSIONS)
            if error_msg:
                return jsonify({"success": False, "error": error_msg})
        else:
            return jsonify({"success": False, "error": "Файл не найден"})

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

        # Очистка памяти после конвертации
        gc.collect()

        return jsonify({"success": True, "output_path": output_path, "download_url": f"/download/{os.path.basename(output_path)}"})

    except Exception as e:
        # Очистка временного файла в случае ошибки
        cleanup_temp_file(input_path)
        return jsonify({"success": False, "error": str(e)})
    except RequestEntityTooLarge:
        cleanup_temp_file(input_path)
        return jsonify({"success": False, "error": "Файл слишком большой (максимум 500MB)"})


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

        # Очистка памяти после конвертации
        gc.collect()

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
    zip_path = None
    try:
        if "model_file" not in request.files:
            return jsonify({"success": False, "error": "Файл не найден"})

        file = request.files["model_file"]
        model_name = request.form.get("model_name")

        if not model_name or model_name.strip() == "":
            return jsonify({"success": False, "error": "Имя модели не указано"})

        # Дополнительная проверка для ZIP файлов
        if not file.filename.lower().endswith(".zip"):
            return jsonify({"success": False, "error": "Требуется ZIP файл"})

        # Используем новую функцию для безопасного сохранения файла
        zip_path, error_msg = save_uploaded_file(file, UPLOAD_FOLDER, {"zip"})
        if error_msg:
            return jsonify({"success": False, "error": error_msg})

        print(f"[DEBUG] Файл сохранен: {zip_path}, существует: {os.path.exists(zip_path)}")
        try:
            print(f"[DEBUG] Начинаем загрузку модели из: {zip_path}")
            result = upload_model_zip(zip_path=zip_path, model_name=model_name)
            print(f"[DEBUG] Загрузка завершена успешно: {result}")

            cleanup_temp_file(zip_path)

            return jsonify({"success": True, "message": result})
        except Exception as upload_error:
            print(f"[ERROR] Ошибка загрузки модели: {upload_error}")
            cleanup_temp_file(zip_path)
            return jsonify({"success": False, "error": str(upload_error)})

    except Exception as e:
        cleanup_temp_file(zip_path)
        return jsonify({"success": False, "error": str(e)})
    except RequestEntityTooLarge:
        cleanup_temp_file(zip_path)
        return jsonify({"success": False, "error": "Файл слишком большой (максимум 500MB)"})


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
        output_dir = "output/RVC_output"
        file_path = os.path.join(output_dir, filename)

        if validate_file_exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "Файл не найден"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "Файл слишком большой (максимум 500MB)", "error_code": "FILE_TOO_LARGE"}), 413


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"success": False, "error": "Файл слишком большой (максимум 500MB)", "error_code": "FILE_TOO_LARGE"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Страница не найдена", "error_code": "NOT_FOUND"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Внутренняя ошибка сервера", "error_code": "INTERNAL_ERROR"}), 500


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"success": False, "error": "Неверный запрос", "error_code": "BAD_REQUEST"}), 400


if __name__ == "__main__":
    import logging

    parser = argparse.ArgumentParser(description="Mushroom RVC Web UI")
    parser.add_argument("--cloudflared", action="store_true", help="Запустить с Cloudflared туннелем")
    parser.add_argument("--localtunnel", action="store_true", help="Запустить с локальным туннелем (localtunnel)")
    parser.add_argument("--lang", choices=["ru", "en"], default="ru", help="Язык интерфейса (ru/en) / Interface language (ru/en)")
    parser.add_argument("--port", type=int, default=5000, help="Порт для веб-сервера")
    parser.add_argument("--host", default="0.0.0.0", help="Хост для веб-сервера")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")

    args = parser.parse_args()

    if not args.debug:
        werkzeug_logger = logging.getLogger("werkzeug")
        werkzeug_logger.setLevel(logging.ERROR)
        werkzeug_logger.disabled = True

    # Настройка локального туннеля если указан флаг
    tunnel_url = None
    if args.localtunnel:
        tunnel_url = setup_localtunnel(args.port)

    print(f"\n🌌 Mushroom RVC WebUI запущен!")
    print(f"📡 Локальный адрес: http://localhost:{args.port}")
    if tunnel_url:
        print(f"🌐 Публичный адрес: {tunnel_url}")
    print(f"🔧 Режим отладки: {'включен' if args.debug else 'отключен'}")
    print(f"\n💡 Для остановки сервера нажмите Ctrl+C\n")

    app.run(debug=args.debug, host=args.host, port=args.port)
