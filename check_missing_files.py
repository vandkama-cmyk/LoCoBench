#!/usr/bin/env python3
"""
Скрипт для проверки отсутствующих файлов в сценариях.
Проходит по директории data/generated и сравнивает файлы с файлами из сценариев.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

def find_scenario_files(scenarios_dir: Path) -> Dict[str, List[str]]:
    """Найти все сценарии и извлечь из них список файлов по проектам"""
    project_files = defaultdict(list)
    
    for scenario_file in scenarios_dir.glob("*.json"):
        try:
            with open(scenario_file, 'r', encoding='utf-8') as f:
                scenario = json.load(f)
            
            scenario_id = scenario.get('id', '')
            context_files = scenario.get('context_files', [])
            
            # Извлечь имя проекта из scenario_id
            # Формат: {project_name_with_number}_{task_category}_{difficulty}_{num}
            # Пример: java_web_ecommerce_expert_036_feature_implementation_hard_01
            # где project_name_with_number = java_web_ecommerce_expert_036
            if scenario_id:
                parts = scenario_id.split('_')
                if len(parts) >= 4:
                    # Найти task_category в частях
                    # Task categories: feature, bug, refactor, security, performance, test
                    task_categories = ['feature', 'bug', 'refactor', 'security', 'performance', 'test']
                    project_name = None
                    
                    # Ищем task_category в частях
                    for i in range(len(parts)):
                        if parts[i] in task_categories and i > 0:
                            # Взять все части до task_category (включая номер проекта)
                            project_name = '_'.join(parts[:i])
                            break
                    
                    # Если task_category не найден, попробуем найти difficulty
                    if not project_name:
                        difficulties = ['easy', 'medium', 'hard', 'expert']
                        for i in range(len(parts)):
                            if parts[i] in difficulties and i > 0:
                                # Проверим, нет ли task_category перед difficulty
                                # Если есть числа перед difficulty, это может быть номер проекта
                                # Возьмем все до difficulty
                                project_name = '_'.join(parts[:i])
                                break
                    
                    # Если все еще не найдено, попробуем найти паттерн с номером в конце
                    # (последние 3 символа перед task_category/difficulty должны быть числа)
                    if not project_name and len(parts) >= 5:
                        # Попробуем найти паттерн: ..._XXX_task_category или ..._XXX_difficulty
                        # где XXX - трехзначное число
                        for i in range(2, len(parts) - 2):
                            # Проверим, является ли текущая часть числом (номер проекта)
                            if parts[i].isdigit() and len(parts[i]) == 3:
                                # Следующая часть должна быть task_category или difficulty
                                if i + 1 < len(parts) and (parts[i + 1] in task_categories or parts[i + 1] in ['easy', 'medium', 'hard', 'expert']):
                                    project_name = '_'.join(parts[:i + 1])
                                    break
                    
                    # Fallback: если ничего не найдено, возьмем больше частей
                    # Обычно проект имеет формат: language_category_type_level_number
                    if not project_name:
                        # Попробуем взять части до первого числового значения (номера сценария)
                        # или до известных ключевых слов
                        for i in range(len(parts)):
                            if parts[i].isdigit() and len(parts[i]) <= 2:  # Номер сценария (01, 02, etc)
                                if i > 0:
                                    project_name = '_'.join(parts[:i])
                                    break
                        
                        # Если все еще не найдено, возьмем первые 5 частей (обычно достаточно)
                        if not project_name:
                            project_name = '_'.join(parts[:min(5, len(parts))])
                    
                    if project_name:
                        if isinstance(context_files, list):
                            project_files[project_name].extend(context_files)
                        elif isinstance(context_files, dict):
                            project_files[project_name].extend(context_files.keys())
                    else:
                        print(f"⚠️ Не удалось извлечь имя проекта из scenario_id: {scenario_id}")
        except Exception as e:
            print(f"Ошибка при чтении {scenario_file}: {e}")
    
    # Удалить дубликаты
    for project in project_files:
        project_files[project] = list(set(project_files[project]))
    
    # Вывести информацию о найденных проектах для отладки
    print(f"\nНайдено проектов в сценариях: {len(project_files)}")
    print("Примеры проектов:")
    for i, project_name in enumerate(list(project_files.keys())[:10], 1):
        print(f"  {i}. {project_name} ({len(project_files[project_name])} файлов)")
    
    return project_files

def find_actual_files(generated_dir: Path, project_name: str) -> Set[str]:
    """Найти все реальные файлы в директории проекта"""
    actual_files = set()
    project_path = generated_dir / project_name
    
    if not project_path.exists():
        return actual_files
    
    # Найти все файлы в проекте (рекурсивно)
    for root, dirs, files in os.walk(project_path):
        root_path = Path(root)
        rel_root = root_path.relative_to(project_path)
        
        for file in files:
            if rel_root == Path('.'):
                file_path = file
            else:
                file_path = rel_root / file
            actual_files.add(str(file_path).replace('\\', '/'))
    
    return actual_files

def normalize_path(path: str, project_dir_name: str = None) -> str:
    """Нормализовать путь, убрав дублирование имен директорий"""
    # Убрать ведущие слэши
    normalized = path.lstrip('/').lstrip('\\')
    
    # Если путь содержит имя проекта, взять все после последнего вхождения с разделителем
    if project_dir_name and project_dir_name in normalized:
        # Найти последнее вхождение project_dir_name с разделителем пути
        # Важно: искать только если за ним следует разделитель, чтобы избежать частичных совпадений
        search_pattern = project_dir_name + '/'
        idx = normalized.rfind(search_pattern)
        if idx == -1:
            search_pattern = project_dir_name + '\\'
            idx = normalized.rfind(search_pattern)
        
        if idx != -1:
            after_project = normalized[idx + len(search_pattern):]
            if after_project:
                normalized = after_project
    
    return normalized

def check_missing_files(generated_dir: Path, scenarios_dir: Path) -> Dict[str, Dict]:
    """Проверить отсутствующие файлы для каждого проекта"""
    results = {}
    
    # Получить список файлов из сценариев
    scenario_files = find_scenario_files(scenarios_dir)
    
    print(f"Найдено {len(scenario_files)} проектов в сценариях")
    
    for project_name, expected_files in scenario_files.items():
        print(f"\nПроверка проекта: {project_name}")
        print(f"  Ожидается файлов: {len(expected_files)}")
        
        # Найти реальные файлы
        actual_files = find_actual_files(generated_dir, project_name)
        print(f"  Найдено файлов: {len(actual_files)}")
        
        # Если директория проекта не найдена, попробовать найти похожие директории
        project_path = generated_dir / project_name
        if not project_path.exists():
            # Попробовать найти директории, которые начинаются с project_name
            potential_dirs = []
            if generated_dir.exists():
                for dir_path in generated_dir.iterdir():
                    if dir_path.is_dir() and dir_path.name.startswith(project_name):
                        potential_dirs.append(dir_path.name)
            
            if potential_dirs:
                print(f"  ⚠️ Директория {project_name} не найдена, но найдены похожие: {potential_dirs[:3]}")
                # Использовать первую найденную директорию
                project_name = potential_dirs[0]
                actual_files = find_actual_files(generated_dir, project_name)
                print(f"  Используется директория: {project_name}, найдено файлов: {len(actual_files)}")
            else:
                print(f"  ⚠️ Директория {project_name} не найдена!")
                results[project_name] = {
                    'total_expected': len(expected_files),
                    'total_found': 0,
                    'total_missing': len(expected_files),
                    'missing_files': [{'expected_path': f, 'normalized_path': f} for f in expected_files[:50]],
                    'missing_count': len(expected_files),
                    'error': 'Project directory not found'
                }
                continue
        
        # Определить имя директории проекта и все возможные поддиректории
        project_path = generated_dir / project_name
        project_dir_name = None
        all_subdirs = []
        
        if project_path.exists():
            # Найти все поддиректории проекта
            subdirs = [d for d in project_path.iterdir() if d.is_dir()]
            if subdirs:
                # Использовать первую поддиректорию как основную
                project_dir_name = subdirs[0].name
                # Сохранить все поддиректории для поиска
                all_subdirs = [d.name for d in subdirs]
            else:
                project_dir_name = project_name
        else:
            project_dir_name = project_name
        
        # Проверить каждый ожидаемый файл
        missing_files = []
        found_files = []
        
        for expected_file in expected_files:
            # Нормализовать путь
            normalized_expected = normalize_path(expected_file, project_dir_name)
            
            # Создать список всех возможных вариантов пути для поиска
            path_attempts = []
            
            # Вариант 1: нормализованный путь (без project_dir_name и ведущих слэшей)
            if normalized_expected and normalized_expected != expected_file:
                path_attempts.append(normalized_expected)
            
            # Вариант 2: оригинальный путь без ведущего слэша
            original_no_slash = expected_file.lstrip('/').lstrip('\\')
            if original_no_slash and original_no_slash not in path_attempts:
                path_attempts.append(original_no_slash)
            
            # Вариант 3: оригинальный путь как есть
            if expected_file not in path_attempts:
                path_attempts.append(expected_file)
            
            # Вариант 4: если путь содержит много компонентов, попробовать последние 2-3
            if '/' in normalized_expected and normalized_expected.count('/') > 2:
                path_parts = normalized_expected.split('/')
                if len(path_parts) >= 2:
                    last_two = '/'.join(path_parts[-2:])
                    if last_two not in path_attempts:
                        path_attempts.append(last_two)
                if len(path_parts) >= 3:
                    last_three = '/'.join(path_parts[-3:])
                    if last_three not in path_attempts:
                        path_attempts.append(last_three)
            
        for expected_file in expected_files:
            # Нормализовать путь
            normalized_expected = normalize_path(expected_file, project_dir_name)
            
            # Создать список всех возможных вариантов пути для поиска
            path_attempts = []
            
            # Вариант 1: нормализованный путь (без project_dir_name и ведущих слэшей)
            if normalized_expected and normalized_expected != expected_file:
                path_attempts.append(normalized_expected)
            
            # Вариант 2: оригинальный путь без ведущего слэша
            original_no_slash = expected_file.lstrip('/').lstrip('\\')
            if original_no_slash and original_no_slash not in path_attempts:
                path_attempts.append(original_no_slash)
            
            # Вариант 3: оригинальный путь как есть
            if expected_file not in path_attempts:
                path_attempts.append(expected_file)
            
            # Вариант 4: если путь содержит много компонентов, попробовать последние 2-3
            if '/' in normalized_expected and normalized_expected.count('/') > 2:
                path_parts = normalized_expected.split('/')
                if len(path_parts) >= 2:
                    last_two = '/'.join(path_parts[-2:])
                    if last_two not in path_attempts:
                        path_attempts.append(last_two)
                if len(path_parts) >= 3:
                    last_three = '/'.join(path_parts[-3:])
                    if last_three not in path_attempts:
                        path_attempts.append(last_three)
                if len(path_parts) >= 4:
                    last_four = '/'.join(path_parts[-4:])
                    if last_four not in path_attempts:
                        path_attempts.append(last_four)
            
            # Попробовать найти файл по всем вариантам путей
            found = False
            for path_attempt in path_attempts:
                if path_attempt in actual_files:
                    found = True
                    found_files.append(expected_file)
                    break
            
            # Если не найдено в списке, проверить напрямую в файловой системе
            if not found:
                project_path = generated_dir / project_name
                if project_path.exists():
                    # Попробовать все варианты путей напрямую в файловой системе
                    for path_attempt in path_attempts:
                        potential_file = project_path / path_attempt
                        if potential_file.exists() and potential_file.is_file():
                            # Найти относительный путь и добавить в actual_files для будущих проверок
                            rel_path = potential_file.relative_to(project_path)
                            rel_path_str = str(rel_path).replace('\\', '/')
                            actual_files.add(rel_path_str)
                            found = True
                            found_files.append(expected_file)
                            break
                    
                    # Если все еще не найдено, попробовать поиск во всех поддиректориях
                    if not found:
                        for subdir in project_path.iterdir():
                            if subdir.is_dir():
                                for path_attempt in path_attempts:
                                    potential_file = subdir / path_attempt
                                    if potential_file.exists() and potential_file.is_file():
                                        rel_path = potential_file.relative_to(project_path)
                                        rel_path_str = str(rel_path).replace('\\', '/')
                                        actual_files.add(rel_path_str)
                                        found = True
                                        found_files.append(expected_file)
                                        break
                                
                                # Также попробовать путь относительно поддиректории
                                if not found and '/' in expected_file:
                                    path_parts = expected_file.split('/', 1)
                                    if len(path_parts) > 1:
                                        potential_file = subdir / path_parts[1]
                                        if potential_file.exists() and potential_file.is_file():
                                            rel_path = potential_file.relative_to(project_path)
                                            rel_path_str = str(rel_path).replace('\\', '/')
                                            actual_files.add(rel_path_str)
                                            found = True
                                            found_files.append(expected_file)
                                            break
                            
                            if found:
                                break
            
            # Если все еще не найдено, попробовать более агрессивный поиск в файловой системе
            if not found:
                project_path = generated_dir / project_name
                if project_path.exists():
                    file_name = Path(expected_file).name
                    
                    # Рекурсивный поиск по имени файла
                    if file_name:
                        matches_found = []
                        for root, dirs, files in os.walk(project_path):
                            if file_name in files:
                                found_path = Path(root) / file_name
                                matches_found.append(found_path)
                        
                        if len(matches_found) == 1:
                            # Одно совпадение - использовать его
                            found_path = matches_found[0]
                            rel_path = found_path.relative_to(project_path)
                            rel_path_str = str(rel_path).replace('\\', '/')
                            actual_files.add(rel_path_str)
                            found = True
                            found_files.append(expected_file)
                        elif len(matches_found) > 1:
                            # Несколько совпадений - попробовать найти по части пути
                            if '/' in expected_file:
                                expected_parts = expected_file.split('/')
                                # Попробовать найти совпадение по последним компонентам
                                for suffix_len in [4, 3, 2, 1]:
                                    if len(expected_parts) >= suffix_len:
                                        expected_suffix = '/'.join(expected_parts[-suffix_len:])
                                        for match_path in matches_found:
                                            match_rel = match_path.relative_to(project_path)
                                            match_str = str(match_rel).replace('\\', '/')
                                            if match_str.endswith(expected_suffix):
                                                actual_files.add(match_str)
                                                found = True
                                                found_files.append(expected_file)
                                                break
                                        if found:
                                            break
                            # Если все еще не найдено и есть только несколько совпадений, использовать первое
                            if not found and len(matches_found) <= 3:
                                found_path = matches_found[0]
                                rel_path = found_path.relative_to(project_path)
                                rel_path_str = str(rel_path).replace('\\', '/')
                                actual_files.add(rel_path_str)
                                found = True
                                found_files.append(expected_file)
            
            
            if not found:
                missing_files.append({
                    'expected_path': expected_file,
                    'normalized_path': normalized_expected,
                    'attempted_paths': path_attempts[:5]  # Сохранить первые 5 попыток для отладки
                })
        
        results[project_name] = {
            'total_expected': len(expected_files),
            'total_found': len(found_files),
            'total_missing': len(missing_files),
            'missing_files': missing_files[:50],  # Ограничить до 50 для читаемости
            'missing_count': len(missing_files)
        }
        
        print(f"  Найдено: {len(found_files)}, Отсутствует: {len(missing_files)}")
    
    return results

def main():
    generated_dir = Path('data/generated')
    scenarios_dir = Path('data/output/scenarios')
    output_file = Path('missing_files_report.json')
    
    if not generated_dir.exists():
        print(f"Директория {generated_dir} не найдена!")
        return
    
    if not scenarios_dir.exists():
        print(f"Директория {scenarios_dir} не найдена!")
        return
    
    print("Начало проверки файлов...")
    results = check_missing_files(generated_dir, scenarios_dir)
    
    # Сохранить результаты
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в {output_file}")
    
    # Вывести статистику
    total_missing = sum(r['missing_count'] for r in results.values())
    total_expected = sum(r['total_expected'] for r in results.values())
    
    print(f"\nОбщая статистика:")
    print(f"  Проектов проверено: {len(results)}")
    print(f"  Всего ожидалось файлов: {total_expected}")
    print(f"  Всего отсутствует файлов: {total_missing}")
    
    # Топ-10 проектов с наибольшим количеством отсутствующих файлов
    sorted_projects = sorted(results.items(), key=lambda x: x[1]['missing_count'], reverse=True)
    print(f"\nТоп-10 проектов с наибольшим количеством отсутствующих файлов:")
    for i, (project, data) in enumerate(sorted_projects[:10], 1):
        print(f"  {i}. {project}: {data['missing_count']}/{data['total_expected']} отсутствует")

if __name__ == '__main__':
    main()
