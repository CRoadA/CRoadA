import sys
import json
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, QObject, pyqtSlot, pyqtSignal
from PyQt6.QtWebChannel import QWebChannel
from collections.abc import Callable
import asyncio
from qasync import asyncSlot

# -------- Python Bridge Object ---------

class WebBridge(QObject):
    geojsonReceived = pyqtSignal(dict)

    @pyqtSlot(str)
    def receiveGeoJSON(self, geojson_str):
        """Receive geoJSON string from JavaScript."""
        geojson = json.loads(geojson_str)
        self.geojsonReceived.emit(geojson)


# -------- Main Window ---------

class MapWindow(QWidget):
    def __init__(self, on_prediction_request : Callable[[dict], None], on_city_location : Callable[[dict], None]):
        super().__init__()
        self.on_prediction_request = on_prediction_request
        self.on_city_location = on_city_location

        # self.setWindowTitle("OpenStreetMap in PyQt + GeoJSON bbox")

        self.browser = QWebEngineView()
        # self.setCentralWidget(self.browser)
        layout = QVBoxLayout()
        layout.addWidget(self.browser)
        self.setLayout(layout)


        # Bridge for communication
        self.bridge = WebBridge()

        # Setup JS channel
        self.channel = QWebChannel(self.browser.page())
        self.channel.registerObject("bridge", self.bridge)
        self.browser.page().setWebChannel(self.channel)

        self.load_page()

        # Storage for latest GeoJSON
        self.current_geojson = None
        self.bridge.geojsonReceived.connect(self._on_geojson)


    @asyncSlot(dict)
    async def _on_geojson(self, geojson):
        """Store and print received GeoJSON."""
        self.current_geojson = geojson
        print("Received GeoJSON:", json.dumps(geojson, indent=2))
        if geojson.get("type") == "Polygon":
            asyncio.create_task(self.on_prediction_request(geojson))
        if geojson.get("type") == "CityRequest":
            task = asyncio.create_task(self.on_city_location(geojson))
            coords = await task
            if coords:
                self.browser.page().runJavaScript(f"displayCoords({coords[0]}, {coords[1]})")



    def get_visible_geojson(self):
        """
        Ask JS for visible map area. 
        Result arrives asynchronously via WebChannel → _on_geojson().
        """
        js = "getVisibleGeoJSON();"
        self.browser.page().runJavaScript(js)

    def load_page(self):
        abs_path = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(abs_path, "templates", "map.html")
        with open(html_path, "r", encoding="UTF-8") as html_file:
            html = html_file.read()
            self.browser.setHtml(html, QUrl("https://localhost/"))

"""         js_path = os.path.join("static", "map.js")
        with open(js_path, "r") as js_file:
            js = js_file.read()
            self.browser.page().runJavaScript(js)
 """
