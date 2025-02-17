# Product Search by Image

## Описание проекта

Этот проект представляет собой веб-приложение для поиска продуктов на основе изображений. Приложение использует предобученную нейронную сеть для извлечения признаков изображений и Django с Django Rest Framework (DRF) для создания RESTful API и пользовательского интерфейса. Интерфейс выполнен в зеленых или синих тонах, что делает его визуально привлекательным и удобным для пользователей.

## Функциональные возможности

- **Загрузка изображений продуктов**: Пользователи могут загружать изображения продуктов, которые затем будут использоваться для поиска.
- **Поиск по изображениям**: Система находит продукты, похожие на загруженное изображение, используя алгоритмы извлечения признаков.
- **Просмотр результатов поиска**: Результаты поиска отображаются с изображениями и информацией о найденных продуктах.
- **Пользовательский интерфейс**: Интерфейс приложения прост в использовании и поддерживает цветовую схему в зеленых или синих тонах для улучшения визуального восприятия.

## Установка и запуск

1. **Клонирование репозитория**:
   Сначала клонируйте репозиторий проекта на свой локальный компьютер с помощью команды `git clone`, затем перейдите в директорию проекта.

2. **Создание виртуального окружения**:
   Создайте виртуальное окружение для изоляции зависимостей проекта.

3. **Активирование виртуального окружения**:
   Активируйте виртуальное окружение в зависимости от вашей операционной системы.

4. **Установка зависимостей**:
   Установите все необходимые зависимости, перечисленные в `requirements.txt`, используя команду `pip install`.

5. **Выполнение миграций базы данных**:
   Примените миграции для настройки базы данных с помощью команды `python manage.py migrate`.

6. **Запуск сервера разработки**:
   Запустите сервер разработки Django командой `python manage.py runserver`. Приложение будет доступно по адресу `http://127.0.0.1:8000/`.

## Реализованные функции

- **API для управления продуктами**:
  Создан API для управления продуктами, включая возможность создания, получения, обновления и удаления записей.

- **Моделирование продуктов**:
  Реализована модель для хранения информации о продуктах, включая название, изображение и описание.

- **Сериализация данных**:
  Реализован сериализатор для преобразования объектов модели продукта в формат JSON и обратно, что позволяет эффективно обмениваться данными между клиентом и сервером.

- **Маршрутизация URL**:
  Настроены маршруты для обработки запросов к ресурсам продуктов с использованием `DefaultRouter`, что упрощает настройку и управление API.

## Используемые технологии

- **Django**: Веб-фреймворк для разработки веб-приложений.
- **Django Rest Framework (DRF)**: Библиотека для создания RESTful API.
- **TensorFlow/PyTorch**: Используются для извлечения признаков из изображений ).
- **База данных**: SQLite в зависимости от конфигурации.

## Контакты

Если у вас есть вопросы или предложения, пожалуйста, свяжитесь со мной по электронной почте: [gulmiraisakbaeva78@gmail.com].

