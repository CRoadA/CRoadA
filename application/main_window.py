from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QWidget, QSplitter, QFrame, QSizePolicy, QScrollArea, QStackedWidget
)

from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from application.MapWindow import MapWindow
from graph_remaker.prediction_statistics import PredictionStatistics
from collections.abc import Callable
import os
import numpy as np
from enum import Enum
import cv2


class RESULTS_TYPE(Enum):
    ERROR = "Error"
    PREDICTION = "Prediction"
    STATUS = "Status"


class MainWindow(QMainWindow):
    def __init__(self, on_save_marked_city: Callable[[dict], None], on_city_location: Callable[[dict], None], 
                 on_handle_prediction: Callable[[None], None], styles : str = "main.css"):
        super().__init__()
        self.setWindowTitle("CRoadA - Urban Analysis")
        self.setMinimumSize(400,500)
        self.showMaximized()
        
        self.initialize_widgets(on_save_marked_city, on_city_location, on_handle_prediction)
        
        self.apply_styles(styles)


    def initialize_widgets(self, on_save_marked_city: Callable[[dict], None], on_city_location : Callable[[dict], None], on_handle_prediction: Callable[[None], None]):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Główny layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar.setFixedWidth(400)
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(15)

        title_label = QLabel("Control Panel")
        title_label.setObjectName("HeaderLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Przycisk
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.predict_btn.setMinimumHeight(50)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setProperty("class", "inactive")
        self.predict_btn.clicked.connect(lambda : self.handle_prediction(on_handle_prediction))

        # Sekcja wyników
        results_frame = QFrame()
        results_frame.setObjectName("ResultsFrame")
        results_layout = QVBoxLayout(results_frame)
        
        self.results_header = QLabel()
        self.results_header.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.results_header.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        results_layout.addWidget(self.results_header)
        results_layout.addWidget(self.result_label)
        results_layout.addStretch()

        sidebar_layout.addWidget(title_label)
        sidebar_layout.addWidget(self.predict_btn)
        sidebar_layout.addWidget(results_frame)
        sidebar_layout.addStretch()


        # Prawa strona
        right_side_widget = QWidget()
        right_side_layout = QVBoxLayout(right_side_widget)
        right_side_layout.setContentsMargins(0, 0, 0, 0)
        right_side_layout.setSpacing(0)

        # Pasek nawigacyjny
        navbar = QFrame()
        navbar.setFixedHeight(50)
        navbar.setObjectName("Navbar")
        navbar_layout = QHBoxLayout(navbar)
        navbar_layout.setContentsMargins(10, 0, 10, 0)
        navbar_layout.setSpacing(10)
        navbar_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.btn_map = QPushButton("Map")
        self.btn_map.setObjectName("NavButton")
        self.btn_map.setCheckable(True)
        self.btn_map.setChecked(True)
        self.btn_map.clicked.connect(self.show_map_view)

        self.btn_result = QPushButton("Results")
        self.btn_result.setObjectName("NavButton")
        self.btn_result.setCheckable(True)
        self.btn_result.clicked.connect(self.show_result_view)
        
        # Notification label
        self.notification_badge = QLabel("1", navbar)
        self.notification_badge.setObjectName("NotificationBadge")
        self.notification_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.notification_badge.hide()
        self.notification_badge.resize(20, 20)
        
        self.btn_result.installEventFilter(self)

        navbar_layout.addWidget(self.btn_map)
        navbar_layout.addWidget(self.btn_result)
        
        self.notification_badge.raise_()

        # Kontener na widoki
        self.stacked_widget = QStackedWidget()

        # Mapa
        self.map_widget = MapWindow(on_save_marked_city, on_city_location)
        self.map_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stacked_widget.addWidget(self.map_widget)

        # Wynik
        self.results_container = QWidget()
        results_view_layout = QVBoxLayout(self.results_container)
        
        self.result_scroll_area = QScrollArea()
        self.result_scroll_area.setWidgetResizable(True)
        self.grid_display_label = QLabel("Brak wyników")
        self.grid_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_scroll_area.setWidget(self.grid_display_label)
        
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.stats_label.setMaximumHeight(150) # Limit height for stats
        
        results_view_layout.addWidget(self.result_scroll_area)
        results_view_layout.addWidget(self.stats_label)
        
        self.stacked_widget.addWidget(self.results_container)

        right_side_layout.addWidget(navbar)
        right_side_layout.addWidget(self.stacked_widget)


        # splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.sidebar)
        splitter.addWidget(right_side_widget)
        
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)


    def eventFilter(self, source, event):
        if source == self.btn_result and event.type() in (event.Type.Resize, event.Type.Move):
            w = self.btn_result.width()      
            new_x = self.btn_result.x() + self.btn_result.width() - 14
            new_y = self.btn_result.y() - 3
            
            self.notification_badge.move(new_x, new_y)
            self.notification_badge.raise_() # Keep on top
            
        return super().eventFilter(source, event)


    def apply_styles(self, styles_file : str):
        abspath = os.path.join(os.path.dirname(__file__), "static")
        try:
            with open(os.path.join(abspath, styles_file), "r", encoding="utf-8") as file:
                styles = file.read()

            self.setStyleSheet(styles)
        except Exception as e:
            print(f"Error while loading styles: {e}")



    def set_results(self, text : str, results_type : RESULTS_TYPE):
        self.results_header.setText(str(results_type.value).replace("(", "").replace(")", ""))  
        self.result_label.setText(text)


    def show_map_view(self):
        self.stacked_widget.setCurrentIndex(0)
        self.btn_map.setChecked(True)
        self.btn_result.setChecked(False)


    def show_result_view(self):
        self.hide_notification()
        self.stacked_widget.setCurrentIndex(1)
        self.btn_map.setChecked(False)
        self.btn_result.setChecked(True)


    def display_grid(self, grid: np.ndarray):
        if grid is None:
            return

        if grid.dtype != np.uint8:
            grid_normalized = cv2.normalize(grid, None, 0, 255, cv2.NORM_MINMAX) if grid.max() > 1 else grid * 255
            grid = grid_normalized.astype(np.uint8)
        
        # konwersja na QImage
        height, width = grid.shape
        bytes_per_line = width
        q_image = QImage(grid.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        
        self.grid_display_label.setPixmap(pixmap)
        self.grid_display_label.setScaledContents(True)

        if self.stacked_widget.currentIndex() != 1:
            self.show_notification()


    def show_notification(self):
        self.notification_badge.show()
        self.notification_badge.raise_()


    def hide_notification(self):
        self.notification_badge.hide()


    def prediction_button_activation(self, activate : bool):
        self.predict_btn.setEnabled(activate)
        if activate:
            self.predict_btn.setProperty("class", "")
        else:
            self.predict_btn.setProperty("class", "inactive")

        self.predict_btn.style().unpolish(self.predict_btn)
        self.predict_btn.style().polish(self.predict_btn)


    def handle_prediction(self, on_handle_prediction: Callable[[None], None]):
        self.prediction_button_activation(False)

        self.thread = QThread()
        self.worker = AsynchronousTaskWrapper(on_handle_prediction)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
    
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()



class AsynchronousTaskWrapper(QObject):
    finished = pyqtSignal()
    
    def __init__(self, processing_function):
        super().__init__()
        self.processing_function = processing_function

    def run(self):
        try:
            self.processing_function()
        except Exception as e:
            print(f"Błąd w wątku predykcji: {e}")
        finally:
            self.finished.emit()