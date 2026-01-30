from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QWidget, QSplitter, QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from application.MapWindow import MapWindow
from collections.abc import Callable
import os
from enum import Enum


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

        # Mapa
        self.map_widget = MapWindow(on_save_marked_city, on_city_location)
        self.map_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.map_widget)
        
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)


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