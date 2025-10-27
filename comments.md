проанализируй все содержимое папки  LoCoBench/data/output/scenarios и напиши какие типы ответов возвращаются, например - markdown файлы только, код только, их комбинация и тд
составь таблицу:
типы возращающихся ответов(как описано выше); уникальные сложности задач для типа ответа; диапазон context_length для типа ответа

Best practice: Hybrid (semantic + graph), test на small subsets.
**Hybrid подход**:

- **Семантический поиск**: Используйте CodeBERT (или OpenAI embeddings через API) для ранжирования файлов по cosine similarity к prompt. Пример: sentence-transformers в Python.
- **Графовый анализ**: Постройте dependency graph (e.g., ast для Python, javaparser для Java) и примените PageRank (библиотека networkx). Комбинируйте: final_score = alpha * semantic_score + (1-alpha) * graph_score (alpha ~0.6–0.7).
- **Интеграция в LoCoBench**: Модифицируйте context_preparation.py, добавив семантический retriever (e.g., HuggingFace model). Сохраняйте top-k файлов (k=10–20 для 10K токенов).


Пайплайн evaluate: шаг за шагом
Команда locobench evaluate --model gpt-4o --config-path config.yaml запускает полный цикл оценки:

Загрузка: Модель (GPT-4o) подключается через API (требует ключ в api.sh).
Подготовка данных: Из data/ (распакованный ZIP) берутся сценарии; контекст фильтруется по релевантности (граф зависимостей, PageRank).
Генерация: Модель генерирует код/анализ для каждой задачи (feature implementation, bug fix и т.д.) с таймаутом 3600 с.
Валидация: Код компилируется/запускается в Docker (языко-специфичные образы, e.g., python:3.11-slim), тесты проходят автоматически.
Агрегация: Результаты сохраняются в evaluation_results/ как JSON и Markdown-саммари.

config.yaml задает фильтры (языки, домены, задачи), параметры модели (max_tokens) и пути к данным. Без правильной настройки (e.g., неверный путь) пайплайн падает.
