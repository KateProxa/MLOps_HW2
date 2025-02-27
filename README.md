
# Домашнее задание №2: Автоматизация администрирования MLOps

**Учебный проект для дисциплины "Автоматизация администрирования MLOps II"**  
**Магистерская программа "Инженерия машинного обучения", 2 курс**

### Цель проекта
Целью данного задания является ознакомление с основами управления данными с помощью DVC, управления экспериментами через MLflow и автоматизации с помощью ClearML. В рамках проекта необходимо интегрировать все три инструмента для создания полного ML-пайплайна, который включает управление данными, отслеживание экспериментов и автоматизацию процесса.

## Описание проекта
Проект разделен на три основные части:

1. **Часть 1: Управление данными с DVC** — включает добавление данных в управление с помощью DVC, настройку удаленного хранилища и интеграцию с GitLab CI/CD для автоматического запуска пайплайна.

2. **Часть 2: Управление экспериментами с MLflow** — управление ML-экспериментами, регистрация метрик и моделей.

3. **Часть 3: Автоматизация экспериментов с ClearML** — настройка ClearML для отслеживания задач, логирования метрик и автоматизации экспериментов.

## Структура проекта

```
├── data/                        # Папка с исходными данными
├── public/                      # Папка для отчета
├── data_preprocessing.py        # Скрипт предобработки данных
├── .gitlab-ci.yml               # Конфигурация GitLab CI/CD
├── dvc.yaml                     # Описание пайплайна DVC
├── report_template.html         # Шаблон отчета
├── run_experiments.py           # Скрипт с экспериментами для MLflow
└── README.md                    # Описание проекта
```
