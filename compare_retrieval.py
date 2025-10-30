#!/usr/bin/env python3
"""
Сравнение результатов оценки модели DeepSeek с retrieval и без него
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from locobench.core.config import Config
from locobench.evaluation.evaluator import Evaluator, run_evaluation

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_evaluation_with_config(config_path: str, retrieval_enabled: bool, output_suffix: str):
    """Запуск оценки с указанной конфигурацией"""
    console.print(Panel.fit(f"🧪 Запуск оценки {'С' if retrieval_enabled else 'БЕЗ'} retrieval", 
                           style="bold blue" if retrieval_enabled else "bold yellow"))
    
    # Загрузить конфигурацию
    config = Config.from_yaml(config_path)
    
    # Настроить retrieval
    config.retrieval.enabled = retrieval_enabled
    if retrieval_enabled:
        config.retrieval.method = "embedding"
        config.retrieval.model_name = "all-MiniLM-L6-v2"
        config.retrieval.top_k = 5
        config.retrieval.difficulties = ["hard", "expert"]
        console.print(f"✅ Retrieval включен: метод={config.retrieval.method}, модель={config.retrieval.model_name}")
    else:
        console.print("❌ Retrieval отключен")
    
    # Модель для оценки
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    # Попробовать найти существующие сценарии
    scenarios_dir = Path(config.data.output_dir) / "scenarios"
    
    if not scenarios_dir.exists():
        console.print(f"⚠️ Директория сценариев не найдена: {scenarios_dir}")
        console.print("💡 Создаю тестовый сценарий...")
        
        # Создать тестовый сценарий
        scenarios_dir.mkdir(parents=True, exist_ok=True)
        test_scenario = create_test_scenario()
        scenario_file = scenarios_dir / "test_scenario.json"
        with open(scenario_file, 'w') as f:
            json.dump(test_scenario, f, indent=2)
        console.print(f"✅ Создан тестовый сценарий: {scenario_file}")
    else:
        # Найти существующие сценарии
        scenario_files = list(scenarios_dir.glob("*.json"))
        if not scenario_files:
            console.print(f"⚠️ Сценарии не найдены в {scenarios_dir}")
            console.print("💡 Создаю тестовый сценарий...")
            test_scenario = create_test_scenario()
            scenario_file = scenarios_dir / "test_scenario.json"
            with open(scenario_file, 'w') as f:
                json.dump(test_scenario, f, indent=2)
            console.print(f"✅ Создан тестовый сценарий: {scenario_file}")
        else:
            console.print(f"✅ Найдено {len(scenario_files)} сценариев")
            scenario_file = scenario_files[0]  # Использовать первый доступный
    
    # Загрузить сценарии
    scenarios = []
    if scenario_file.exists():
        with open(scenario_file, 'r') as f:
            scenario_data = json.load(f)
            # Проверить формат
            if isinstance(scenario_data, dict):
                if 'scenarios' in scenario_data:
                    scenarios = scenario_data['scenarios']
                else:
                    scenarios = [scenario_data]  # Один сценарий
            elif isinstance(scenario_data, list):
                scenarios = scenario_data
    
    if not scenarios:
        console.print("❌ Не удалось загрузить сценарии")
        return None
    
    console.print(f"📋 Загружено {len(scenarios)} сценариев")
    
    # Ограничить количество сценариев для теста (первые 2-3)
    test_scenarios = scenarios[:2] if len(scenarios) >= 2 else scenarios
    console.print(f"🎯 Тестирую на {len(test_scenarios)} сценариях")
    
    # Создать evaluator
    evaluator = Evaluator(config, model_name=model_name)
    
    # Запустить оценку
    try:
        console.print(f"\n🤖 Оценка модели: {model_name}")
        console.print(f"📊 Сценариев: {len(test_scenarios)}")
        
        results = await evaluator.evaluate_models(
            model_names=[model_name],
            scenarios=test_scenarios,
            resume=False  # Начать заново для чистого сравнения
        )
        
        # Сохранить результаты
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"deepseek_evaluation_{output_suffix}_{timestamp}.json"
        
        # Преобразовать результаты в JSON-совместимый формат
        results_dict = {}
        for model_name_key, model_results in results.items():
            results_dict[model_name_key] = [
                {
                    'model_name': r.model_name,
                    'scenario_id': r.scenario_id,
                    'total_score': r.total_score,
                    'parsing_success': r.parsing_success,
                    'generation_time': r.generation_time,
                    'software_engineering_score': r.software_engineering_score,
                    'functional_correctness_score': r.functional_correctness_score,
                    'code_quality_score': r.code_quality_score,
                    'longcontext_utilization_score': r.longcontext_utilization_score,
                }
                for r in model_results
            ]
        
        with open(output_file, 'w') as f:
            json.dump({
                'retrieval_enabled': retrieval_enabled,
                'model': model_name,
                'scenarios_count': len(test_scenarios),
                'results': results_dict,
                'timestamp': timestamp
            }, f, indent=2)
        
        console.print(f"✅ Результаты сохранены: {output_file}")
        
        # Создать summary
        summaries = evaluator.generate_evaluation_summary(results)
        
        return {
            'results': results,
            'summaries': summaries,
            'retrieval_enabled': retrieval_enabled,
            'output_file': output_file
        }
        
    except Exception as e:
        console.print(f"❌ Ошибка при оценке: {e}", style="bold red")
        import traceback
        console.print(traceback.format_exc())
        return None


def create_test_scenario():
    """Создать тестовый сценарий для оценки"""
    return {
        "id": "test_python_easy_001",
        "title": "Calculate Factorial Function",
        "description": "Implement a factorial function in Python",
        "difficulty": "easy",
        "task_category": "feature_implementation",
        "language": "python",
        "task_prompt": "Write a Python function `factorial(n)` that calculates the factorial of a number n. The function should handle edge cases (n=0, n=1) and use recursion.",
        "context_files": {
            "utils.py": """
def calculate_sum(a, b):
    '''Calculate sum of two numbers'''
    return a + b

def calculate_product(a, b):
    '''Calculate product of two numbers'''
    return a * b
"""
        },
        "metadata": {
            "project_path": "./data/generated/test_project",
            "context_length": 500
        }
    }


async def compare_results():
    """Сравнить результаты с retrieval и без него"""
    console.print(Panel.fit("🔬 Сравнение оценки DeepSeek с retrieval и без него", style="bold cyan"))
    
    config_path = "config.yaml"
    
    # Запуск 1: БЕЗ retrieval
    console.print("\n" + "="*60)
    results_without = await run_evaluation_with_config(config_path, retrieval_enabled=False, output_suffix="no_retrieval")
    
    console.print("\n" + "="*60)
    console.print("⏳ Ожидание перед вторым запуском...")
    await asyncio.sleep(2)
    
    # Запуск 2: С retrieval
    console.print("\n" + "="*60)
    results_with = await run_evaluation_with_config(config_path, retrieval_enabled=True, output_suffix="with_retrieval")
    
    # Сравнение результатов
    console.print("\n" + "="*60)
    console.print(Panel.fit("📊 Сравнение результатов", style="bold green"))
    
    if results_without and results_with:
        # Извлечь summaries
        summary_without = results_without['summaries']
        summary_with = results_with['summaries']
        
        # Создать таблицу сравнения
        comparison_table = Table(title="Сравнение результатов")
        comparison_table.add_column("Метрика", style="bold")
        comparison_table.add_column("БЕЗ retrieval", style="yellow")
        comparison_table.add_column("С retrieval", style="green")
        comparison_table.add_column("Разница", style="cyan")
        
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        
        if model_name in summary_without and model_name in summary_with:
            s_without = summary_without[model_name]
            s_with = summary_with[model_name]
            
            # Общий счет
            comparison_table.add_row(
                "Общий счет (LCBS)",
                f"{s_without.avg_total_score:.3f}",
                f"{s_with.avg_total_score:.3f}",
                f"{s_with.avg_total_score - s_without.avg_total_score:+.3f}"
            )
            
            # Software Engineering
            comparison_table.add_row(
                "Software Engineering",
                f"{s_without.avg_software_engineering_score:.3f}",
                f"{s_with.avg_software_engineering_score:.3f}",
                f"{s_with.avg_software_engineering_score - s_without.avg_software_engineering_score:+.3f}"
            )
            
            # Functional Correctness
            comparison_table.add_row(
                "Functional Correctness",
                f"{s_without.avg_functional_correctness_score:.3f}",
                f"{s_with.avg_functional_correctness_score:.3f}",
                f"{s_with.avg_functional_correctness_score - s_without.avg_functional_correctness_score:+.3f}"
            )
            
            # Code Quality
            comparison_table.add_row(
                "Code Quality",
                f"{s_without.avg_code_quality_score:.3f}",
                f"{s_with.avg_code_quality_score:.3f}",
                f"{s_with.avg_code_quality_score - s_without.avg_code_quality_score:+.3f}"
            )
            
            # Long-Context Utilization
            comparison_table.add_row(
                "Long-Context Utilization",
                f"{s_without.avg_longcontext_utilization_score:.3f}",
                f"{s_with.avg_longcontext_utilization_score:.3f}",
                f"{s_with.avg_longcontext_utilization_score - s_without.avg_longcontext_utilization_score:+.3f}"
            )
            
            # Parsing Success Rate
            comparison_table.add_row(
                "Parsing Success Rate",
                f"{s_without.parsing_success_rate:.1%}",
                f"{s_with.parsing_success_rate:.1%}",
                f"{s_with.parsing_success_rate - s_without.parsing_success_rate:+.1%}"
            )
            
            # Average Generation Time
            comparison_table.add_row(
                "Avg Generation Time (s)",
                f"{s_without.avg_generation_time:.2f}",
                f"{s_with.avg_generation_time:.2f}",
                f"{s_with.avg_generation_time - s_without.avg_generation_time:+.2f}"
            )
            
            # Completed Scenarios
            comparison_table.add_row(
                "Completed Scenarios",
                f"{s_without.completed_scenarios}/{s_without.total_scenarios}",
                f"{s_with.completed_scenarios}/{s_with.total_scenarios}",
                f"{s_with.completed_scenarios - s_without.completed_scenarios:+d}"
            )
            
            console.print(comparison_table)
            
            # Сохранить сравнение
            comparison_file = Path("evaluation_results") / "comparison_summary.json"
            with open(comparison_file, 'w') as f:
                json.dump({
                    'model': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'without_retrieval': {
                        'avg_total_score': s_without.avg_total_score,
                        'avg_software_engineering': s_without.avg_software_engineering_score,
                        'avg_functional_correctness': s_without.avg_functional_correctness_score,
                        'avg_code_quality': s_without.avg_code_quality_score,
                        'avg_longcontext_utilization': s_without.avg_longcontext_utilization_score,
                        'parsing_success_rate': s_without.parsing_success_rate,
                        'avg_generation_time': s_without.avg_generation_time,
                        'completed_scenarios': s_without.completed_scenarios,
                        'total_scenarios': s_without.total_scenarios
                    },
                    'with_retrieval': {
                        'avg_total_score': s_with.avg_total_score,
                        'avg_software_engineering': s_with.avg_software_engineering_score,
                        'avg_functional_correctness': s_with.avg_functional_correctness_score,
                        'avg_code_quality': s_with.avg_code_quality_score,
                        'avg_longcontext_utilization': s_with.avg_longcontext_utilization_score,
                        'parsing_success_rate': s_with.parsing_success_rate,
                        'avg_generation_time': s_with.avg_generation_time,
                        'completed_scenarios': s_with.completed_scenarios,
                        'total_scenarios': s_with.total_scenarios
                    },
                    'differences': {
                        'total_score_diff': s_with.avg_total_score - s_without.avg_total_score,
                        'parsing_success_diff': s_with.parsing_success_rate - s_without.parsing_success_rate,
                        'generation_time_diff': s_with.avg_generation_time - s_without.avg_generation_time
                    }
                }, f, indent=2)
            
            console.print(f"\n✅ Сравнение сохранено: {comparison_file}")
            
        else:
            console.print("⚠️ Не удалось найти результаты для сравнения")
    else:
        console.print("⚠️ Один или оба запуска завершились с ошибкой")
        if not results_without:
            console.print("❌ Запуск БЕЗ retrieval не удался")
        if not results_with:
            console.print("❌ Запуск С retrieval не удался")


async def main():
    """Главная функция"""
    await compare_results()


if __name__ == "__main__":
    asyncio.run(main())
