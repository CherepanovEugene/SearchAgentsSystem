import logging
import os
import requests
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# 📌 Создаём папку logs, если её нет
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


# 📌 Функция настройки логирования
def setup_logger(agent_name):
    logger = logging.getLogger(agent_name)
    logger.setLevel(logging.INFO)

    log_file = os.path.join(LOG_DIR, f"{agent_name}.log")
    handler = logging.FileHandler(log_file)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# 🟢 Агент коммуникации: получает запрос от пользователя
def communication_agent(user_id, query):
    logger = setup_logger("CommunicationAgent")
    logger.info(f"Получен запрос от пользователя {user_id}: {query}")

    # Проверяем, загружен ли API-ключ
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("API-ключ для OpenRouter.ai не найден!")
        return "Ошибка: API-ключ не найден."

    # Передаём запрос в каждый LLM-агент
    logger.info("Запускаем агенты LLM...")
    perplexity_response = perplexity_agent(query)
    qwen_response = qwen_agent(query)
    deepseek_response = deepseek_agent(query)
    openai_response = openai_agent(query)

    llm_responses = {
        "Perplexity": perplexity_response,
        "Qwen": qwen_response,
        "DeepSeek": deepseek_response,
        "OpenAI GPT-4": openai_response
    }

    logger.info(f"Ответы от LLM: {llm_responses}")

    # Передаём ответы агенту-суммаризатору
    summary = summarization_agent(llm_responses)
    logger.info(f"Суммаризированный ответ: {summary}")

    return summary


# 🔵 Агент для Perplexity
def perplexity_agent(query):
    logger = setup_logger("PerplexityAgent")
    logger.info("Запуск агента Perplexity...")
    return request_openrouter(query, "perplexity/sonar-reasoning", logger)


# 🔵 Агент для Qwen
def qwen_agent(query):
    logger = setup_logger("QwenAgent")
    logger.info("Запуск агента Qwen...")
    return request_openrouter(query, "qwen/qwen-vl-plus:free", logger)


# 🔵 Агент для DeepSeek
def deepseek_agent(query):
    logger = setup_logger("DeepSeekAgent")
    logger.info("Запуск агента DeepSeek...")
    return request_openrouter(query, "deepseek/deepseek-r1-distill-llama-70b:free", logger)


# 🔵 Агент для OpenAI GPT-4
def openai_agent(query):
    logger = setup_logger("OpenAIAgent")
    logger.info("Запуск агента OpenAI GPT-4...")
    return request_openrouter(query, "openai/gpt-4-turbo", logger)


# 🔄 Универсальная функция для запросов в OpenRouter.ai
def request_openrouter(query, model_name, logger):
    api_key = os.getenv("OPENROUTER_API_KEY")
    api_url = "https://openrouter.ai/api/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": query}]
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response_json = response.json()

        # Проверяем, есть ли ошибка в ответе
        if "error" in response_json:
            logger.error(f"Ошибка OpenRouter: {response_json['error']}")
            return "Ошибка API"

        response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "Нет ответа")
        logger.info(f"Ответ от {model_name}: {response_text}")
        return response_text

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка сети при запросе к {model_name}: {str(e)}")
        return "Ошибка сети"


# 🟠 Агент-суммаризатор: делает суммаризацию ответов
def summarization_agent(llm_responses):
    logger = setup_logger("SummarizationAgent")
    logger.info("Запуск суммаризатора...")

    combined_text = "\n".join([f"{model}: {response}" for model, response in llm_responses.items()])

    return request_openrouter(f"Суммаризируй следующее: {combined_text}", "openai/gpt-4-turbo", logger)


# ✅ Тестирование системы
if __name__ == "__main__":
    user_id = "12345"
    query = "Как искусственный интеллект изменит мир в ближайшие 10 лет?"
    result = communication_agent(user_id, query)
    print("\n--- Итоговый ответ ---")
    print(result)