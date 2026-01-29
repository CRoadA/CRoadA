from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QWidget, QSplitter, QFrame, QSizePolicy, QScrollArea
)
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
    def __init__(self, on_prediction_request: Callable[[dict], None], on_city_location: Callable[[dict], None], styles : str = "main.css"):
        super().__init__()
        self.setWindowTitle("CRoadA - Urban Analysis")
        self.setMinimumSize(400,500)
        self.showMaximized()
        
        self.initialize_widgets(on_prediction_request, on_city_location)
        self.apply_styles(styles)


    def initialize_widgets(self, on_prediction_request: Callable[[dict], None], on_city_location : Callable[[dict], None]):
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
        self.map_widget = MapWindow(on_prediction_request, on_city_location)
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