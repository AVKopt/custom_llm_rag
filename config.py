import os

# random seed
RANDOM = 12345
# рабочая директория проекта
WORK_DIR = os.path.abspath(os.curdir)
# директория с LLM GGUF моделями
MODELS_DIR = os.path.join(WORK_DIR, "model")
# директория с эмбеддинг моделью
EMB_MODEL_DIR = os.path.join(WORK_DIR, "emb_model")
# файл модели
LLM_MODEL_NAME = "model-q4_K.gguf"
# модель для эмбеддингов
EMB_MODEL_NAME_LaBSE = "sentence-transformers/LaBSE"
EMB_MODEL_NAME_BGE_M3 = "deepvk/USER-bge-m3"

template = """
Следующее является содержимым контента, релевантным для ответа на вопрос:

{page_content}

Ответ необходимо искать только в 

{page_content}

Используя информацию, предоставленную в содержимом контента, пожалуйста, 
предоставьте подробный и точный ответ на следующий вопрос. 
Ваш ответ должен быть максимально близок к исходному тексту контента, 
сохраняя структуру предложений, лексику и синтаксис. 
Не допускайте расшифровки аббревиатур или перестановки слов.

{quest}

Ответ:
"""

MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]