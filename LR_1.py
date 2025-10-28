import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
import sqlite3
from datetime import datetime
import logging
import os


# Настройка логирования
def setup_logging():
    """Настройка системы логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app_log.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class DataAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.logger = setup_logging()  # Инициализация логгера
        self.initUI()
        self.log_action("Приложение запущено")
        self.logger.info("Приложение инициализировано")

    def initUI(self):
        self.setWindowTitle('Анализатор данных: Ментальное здоровье и соцсети')
        self.setGeometry(100, 100, 1200, 800)

        # Создание вкладок
        self.tabs = QTabWidget()

        # Вкладки
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()

        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()
        self.setup_tab5()

        self.tabs.addTab(self.tab1, "Статистика")
        self.tabs.addTab(self.tab2, "Корреляции")
        self.tabs.addTab(self.tab3, "Тепловая карта")
        self.tabs.addTab(self.tab4, "Линейные графики")
        self.tabs.addTab(self.tab5, "Ход работы")

        self.setCentralWidget(self.tabs)

    def setup_tab1(self):
        layout = QVBoxLayout()

        # Панель загрузки данных
        load_layout = QHBoxLayout()
        self.load_btn = QPushButton('Загрузить CSV файл')
        self.load_btn.clicked.connect(self.load_csv)
        load_layout.addWidget(self.load_btn)

        self.info_label = QLabel('Файл не загружен')
        load_layout.addWidget(self.info_label)
        load_layout.addStretch()

        layout.addLayout(load_layout)

        # Статистика
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(QLabel('Статистика данных:'))
        layout.addWidget(self.stats_text)

        # Превью данных
        self.data_table = QTableWidget()
        layout.addWidget(QLabel('Превью данных (первые 20 строк):'))
        layout.addWidget(self.data_table)

        self.tab1.setLayout(layout)

    def setup_tab2(self):
        layout = QVBoxLayout()

        self.corr_btn = QPushButton('Построить графики корреляции')
        self.corr_btn.clicked.connect(self.plot_correlations)
        layout.addWidget(self.corr_btn)

        self.corr_canvas = MplCanvas(self, width=10, height=8)
        layout.addWidget(self.corr_canvas)

        self.tab2.setLayout(layout)

    def setup_tab3(self):
        layout = QVBoxLayout()

        self.heatmap_btn = QPushButton('Построить тепловую карту')
        self.heatmap_btn.clicked.connect(self.plot_heatmap)
        layout.addWidget(self.heatmap_btn)

        self.heatmap_canvas = MplCanvas(self, width=8, height=6)
        layout.addWidget(self.heatmap_canvas)

        self.tab3.setLayout(layout)

    def setup_tab4(self):
        layout = QVBoxLayout()

        # Выбор столбца
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel('Выберите столбец:'))

        self.column_combo = QComboBox()
        control_layout.addWidget(self.column_combo)

        self.plot_btn = QPushButton('Построить график')
        self.plot_btn.clicked.connect(self.plot_line_chart)
        control_layout.addWidget(self.plot_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        self.line_canvas = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.line_canvas)

        self.tab4.setLayout(layout)

    def setup_tab5(self):
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel('История действий:'))
        layout.addWidget(self.log_text)

        # Кнопки управления логом
        btn_layout = QHBoxLayout()
        self.clear_log_btn = QPushButton('Очистить лог')
        self.clear_log_btn.clicked.connect(self.clear_log)
        btn_layout.addWidget(self.clear_log_btn)

        self.save_log_btn = QPushButton('Сохранить лог')
        self.save_log_btn.clicked.connect(self.save_log)
        btn_layout.addWidget(self.save_log_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.tab5.setLayout(layout)

    def log_action(self, action):
        """Добавление действия в лог"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {action}"
        self.log_text.append(log_entry)
        self.logger.info(action)  # Запись в файл лога

    def clear_log(self):
        """Очистка лога в интерфейсе"""
        self.log_text.clear()
        self.log_action("Лог очищен")
        self.logger.info("Лог интерфейса очищен")

    def save_log(self):
        """Сохранение лога в файл"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить лог", "log_actions.txt", "Text Files (*.txt)"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log_action(f"Лог сохранен в: {filename}")
                self.logger.info(f"Лог действий сохранен в файл: {filename}")
                QMessageBox.information(self, "Успех", "Лог успешно сохранен")
            except Exception as e:
                error_msg = f"Ошибка сохранения лога: {str(e)}"
                self.logger.error(error_msg)
                QMessageBox.critical(self, "Ошибка", error_msg)

    def load_csv(self):
        """Загрузка CSV файла"""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Выберите CSV файл", "", "CSV Files (*.csv)"
            )
            if filename:
                self.logger.info(f"Попытка загрузки файла: {filename}")
                self.df = pd.read_csv(filename)
                self.log_action(f"CSV файл загружен: {os.path.basename(filename)}")
                self.logger.info(f"Файл успешно загружен. Размер: {self.df.shape}")
                self.update_ui_after_data_load()

                QMessageBox.information(
                    self, "Успех",
                    f"Данные загружены!\n"
                    f"Строк: {self.df.shape[0]}, Столбцов: {self.df.shape[1]}"
                )

        except Exception as e:
            error_msg = f"Ошибка загрузки CSV: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)

    def update_ui_after_data_load(self):
        """Обновление интерфейса после загрузки данных"""
        if self.df is not None:
            self.info_label.setText(f"Загружено: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
            self.update_stats()
            self.update_data_table()
            self.update_comboboxes()
            self.logger.info("Интерфейс обновлен после загрузки данных")

    def update_stats(self):
        """Обновление статистики"""
        try:
            stats = []
            stats.append("=== ОСНОВНАЯ СТАТИСТИКА ===\n")
            stats.append(f"Всего записей: {len(self.df)}")
            stats.append(f"Всего столбцов: {len(self.df.columns)}")

            stats.append("\n=== ИНФОРМАЦИЯ О СТОЛБЦАХ ===")
            for col in self.df.columns:
                stats.append(f"\n{col}:")
                stats.append(f"  Тип: {self.df[col].dtype}")
                stats.append(f"  Пустых значений: {self.df[col].isnull().sum()}")
                if self.df[col].dtype in ['int64', 'float64']:
                    stats.append(f"  Мин: {self.df[col].min():.2f}")
                    stats.append(f"  Макс: {self.df[col].max():.2f}")
                    stats.append(f"  Среднее: {self.df[col].mean():.2f}")
                    stats.append(f"  Стандартное отклонение: {self.df[col].std():.2f}")
                else:
                    stats.append(f"  Уникальных значений: {self.df[col].nunique()}")
                    if self.df[col].nunique() <= 10:  # Показываем значения для категориальных с малым числом уникальных
                        stats.append(f"  Значения: {', '.join(map(str, self.df[col].unique()))}")

            self.stats_text.setText('\n'.join(stats))
            self.logger.info("Статистика данных обновлена")

        except Exception as e:
            error_msg = f"Ошибка при расчете статистики: {str(e)}"
            self.logger.error(error_msg)

    def update_data_table(self):
        """Обновление таблицы с данными"""
        try:
            self.data_table.setRowCount(min(20, len(self.df)))
            self.data_table.setColumnCount(len(self.df.columns))
            self.data_table.setHorizontalHeaderLabels(self.df.columns)

            for i in range(min(20, len(self.df))):
                for j in range(len(self.df.columns)):
                    item = QTableWidgetItem(str(self.df.iloc[i, j]))
                    self.data_table.setItem(i, j)

            # Автоподбор ширины столбцов
            self.data_table.resizeColumnsToContents()
            self.logger.info("Таблица данных обновлена")

        except Exception as e:
            error_msg = f"Ошибка при обновлении таблицы: {str(e)}"
            self.logger.error(error_msg)

    def update_comboboxes(self):
        """Обновление выпадающих списков"""
        try:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.column_combo.clear()
            for column in numeric_columns:
                self.column_combo.addItem(column)

            self.logger.info(f"Обновлен список числовых столбцов: {numeric_columns}")

        except Exception as e:
            error_msg = f"Ошибка при обновлении выпадающих списков: {str(e)}"
            self.logger.error(error_msg)

    def plot_correlations(self):
        """Построение графиков корреляции"""
        if self.df is None:
            msg = "Попытка построить график без загруженных данных"
            self.logger.warning(msg)
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        try:
            numeric_df = self.df.select_dtypes(include=[np.number])

            # Очищаем текущую фигуру
            self.corr_canvas.fig.clear()

            # Создаем subplots на текущей фигуре
            fig = self.corr_canvas.fig
            axes = fig.subplots(2, 2)
            fig.suptitle('Корреляции между параметрами ментального здоровья', fontsize=16)

            # График 1: Время у экрана vs Уровень стресса
            axes[0, 0].scatter(numeric_df['Daily_Screen_Time(hrs)'],
                               numeric_df['Stress_Level(1-10)'], alpha=0.6, color='red')
            axes[0, 0].set_xlabel('Ежедневное время у экрана (часы)')
            axes[0, 0].set_ylabel('Уровень стресса (1-10)')
            axes[0, 0].set_title('Время у экрана vs Стресс')
            axes[0, 0].grid(True, alpha=0.3)

            # График 2: Качество сна vs Индекс счастья
            axes[0, 1].scatter(numeric_df['Sleep_Quality(1-10)'],
                               numeric_df['Happiness_Index(1-10)'], alpha=0.6, color='green')
            axes[0, 1].set_xlabel('Качество сна (1-10)')
            axes[0, 1].set_ylabel('Индекс счастья (1-10)')
            axes[0, 1].set_title('Качество сна vs Счастье')
            axes[0, 1].grid(True, alpha=0.3)

            # График 3: Дни без соцсетей vs Уровень стресса
            axes[1, 0].scatter(numeric_df['Days_Without_Social_Media'],
                               numeric_df['Stress_Level(1-10)'], alpha=0.6, color='blue')
            axes[1, 0].set_xlabel('Дни без социальных сетей')
            axes[1, 0].set_ylabel('Уровень стресса (1-10)')
            axes[1, 0].set_title('Дни без соцсетей vs Стресс')
            axes[1, 0].grid(True, alpha=0.3)

            # График 4: Частота упражнений vs Индекс счастья
            axes[1, 1].scatter(numeric_df['Exercise_Frequency(week)'],
                               numeric_df['Happiness_Index(1-10)'], alpha=0.6, color='orange')
            axes[1, 1].set_xlabel('Частота упражнений в неделю')
            axes[1, 1].set_ylabel('Индекс счастья (1-10)')
            axes[1, 1].set_title('Упражнения vs Счастье')
            axes[1, 1].grid(True, alpha=0.3)

            # Автоматическая настройка layout
            fig.tight_layout()

            # Обновляем канвас
            self.corr_canvas.draw()

            self.log_action("Построены графики корреляции")
            self.logger.info("Графики корреляции построены успешно")

        except Exception as e:
            error_msg = f"Ошибка при построении графиков корреляции: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)

    def plot_heatmap(self):
        """Построение тепловой карты корреляций"""
        if self.df is None:
            self.logger.warning("Попытка построить тепловую карту без загруженных данных")
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        try:
            numeric_df = self.df.select_dtypes(include=[np.number])

            self.heatmap_canvas.fig.clear()

            corr_matrix = numeric_df.corr()

            ax = self.heatmap_canvas.fig.add_subplot(111)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                        square=True, cbar_kws={"shrink": .8}, fmt='.2f')
            ax.set_title('Тепловая карта корреляций числовых параметров\n(Ментальное здоровье и соцсети)')

            # Поворот подписей для лучшей читаемости
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)

            self.heatmap_canvas.fig.tight_layout()
            self.heatmap_canvas.draw()

            self.log_action("Построена тепловая карта корреляций")
            self.logger.info("Тепловая карта корреляций построена успешно")

        except Exception as e:
            error_msg = f"Ошибка при построении тепловой карты: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)

    def plot_line_chart(self):
        """Построение линейного графика"""
        if self.df is None:
            self.logger.warning("Попытка построить линейный график без загруженных данных")
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        selected_column = self.column_combo.currentText()
        if not selected_column:
            self.logger.warning("Не выбран столбец для построения графика")
            QMessageBox.warning(self, "Внимание", "Выберите столбец для графика")
            return

        try:
            self.line_canvas.fig.clear()

            ax = self.line_canvas.fig.add_subplot(111)

            # Используем исходный порядок данных для временных рядов
            data = self.df[selected_column]

            ax.plot(data.index, data.values, linewidth=1, marker='o', markersize=2, alpha=0.7)
            ax.set_title(f'Линейный график: {selected_column}')
            ax.set_xlabel('Индекс наблюдения')
            ax.set_ylabel(selected_column)
            ax.grid(True, alpha=0.3)

            # Добавляем статистику на график
            mean_val = data.mean()
            ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.8,
                       label=f'Среднее: {mean_val:.2f}')
            ax.legend()

            self.line_canvas.fig.tight_layout()
            self.line_canvas.draw()

            self.log_action(f"Построен линейный график для: {selected_column}")
            self.logger.info(f"Линейный график построен для столбца: {selected_column}")

        except Exception as e:
            error_msg = f"Ошибка при построении линейного графика: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)


def main():
    """Главная функция приложения"""
    try:
        app = QApplication(sys.argv)

        # Установка стиля для лучшего внешнего вида
        app.setStyle('Fusion')

        window = DataAnalyzerApp()
        window.show()

        logger = logging.getLogger(__name__)
        logger.info("Приложение успешно запущено")

        return app.exec()

    except Exception as e:
        logging.error(f"Критическая ошибка при запуске приложения: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())