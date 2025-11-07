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
            # Формат: {project_name}_{task_category}_{difficulty}_{num}
            if scenario_id:
                parts = scenario_id.split('_')
                if len(parts) >= 4:
                    # Найти первую часть, которая соответствует проекту
                    # Обычно это первые несколько частей до task_category
                    project_name = None
                    task_categories = ['feature', 'bug', 'refactor', 'security', 'performance', 'test']
                    
                    for i in range(len(parts)):
                        if parts[i] in task_categories and i > 0:
                            project_name = '_'.join(parts[:i])
                            break
                    
                    if not project_name:
                        # Fallback: взять первые части
                        project_name = '_'.join(parts[:min(4, len(parts))])
                    
                    if isinstance(context_files, list):
                        project_files[project_name].extend(context_files)
                    elif isinstance(context_files, dict):
                        project_files[project_name].extend(context_files.keys())
        except Exception as e:
            print(f"Ошибка при чтении {scenario_file}: {e}")
    
    # Удалить дубликаты
    for project in project_files:
        project_files[project] = list(set(project_files[project]))
    
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
        
        # Определить имя директории проекта
        project_path = generated_dir / project_name
        project_dir_name = None
        if project_path.exists():
            # Найти поддиректорию проекта
            subdirs = [d for d in project_path.iterdir() if d.is_dir()]
            if subdirs:
                project_dir_name = subdirs[0].name
            else:
                project_dir_name = project_name
        else:
            project_dir_name = project_name
        
        # Проверить каждый ожидаемый файл
        missing_files = []
        found_files = []
        
        for expected_file in expected_files:
            normalized_expected = normalize_path(expected_file, project_dir_name)
            
            # Попробовать найти файл разными способами
            found = False
            
            # Вариант 1: нормализованный путь
            if normalized_expected in actual_files:
                found = True
                found_files.append(expected_file)
                continue
            
            # Вариант 2: оригинальный путь
            if expected_file in actual_files:
                found = True
                found_files.append(expected_file)
                continue
            
            # Вариант 3: путь без ведущего слэша
            no_slash = expected_file.lstrip('/').lstrip('\\')
            if no_slash in actual_files:
                found = True
                found_files.append(expected_file)
                continue
            
            # Вариант 4: поиск по имени файла (только если имя уникальное)
            file_name = Path(expected_file).name
            if file_name and len(file_name) > 10:  # Проверка на уникальность
                matching_files = [f for f in actual_files if f.endswith('/' + file_name) or f == file_name]
                if len(matching_files) == 1:
                    found = True
                    found_files.append(expected_file)
                    continue
            
            # Вариант 5: поиск в поддиректориях, если путь начинается с имени поддиректории
            if not found and '/' in expected_file:
                path_parts = expected_file.split('/', 1)
                if len(path_parts) > 1:
                    potential_subdir = path_parts[0]
                    # Попробовать найти поддиректорию, которая начинается с potential_subdir
                    for actual_file in actual_files:
                        if actual_file.startswith(potential_subdir + '/') or actual_file.startswith(potential_subdir + '\\'):
                            # Проверить, совпадает ли остальная часть пути
                            if '/' in actual_file:
                                actual_parts = actual_file.split('/', 1)
                                if len(actual_parts) > 1 and actual_parts[1] == path_parts[1]:
                                    found = True
                                    found_files.append(expected_file)
                                    break
            
            if not found:
                missing_files.append({
                    'expected_path': expected_file,
                    'normalized_path': normalized_expected
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
