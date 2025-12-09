import sys
import json
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, QObject, pyqtSlot, pyqtSignal
from PyQt6.QtWebChannel import QWebChannel


HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>PyQt OSM Map</title>
<meta name="viewport" content="initial-scale=1.0">
<style>
  html, body, #map { height: 100%; margin: 0; padding: 0; }
</style>

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<!-- WebChannel -->
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>

</head>
<body>

<div id="map"></div>

<script>

var pyObj = null;

// WebChannel connection
new QWebChannel(qt.webChannelTransport, function(channel) {
    pyObj = channel.objects.bridge;
});

var map = L.map('map').setView([51.505, -0.09], 13);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors"
}).addTo(map);

L.marker([51.505, -0.09])
  .addTo(map)
  .bindPopup("Hello from PyQt + OSM!");


// Called by Python — returns GeoJSON bbox of visible area
function getVisibleGeoJSON() {
    let bounds = map.getBounds();

    let geojson = {
        "type": "Polygon",
        "coordinates": [[
            [bounds.getWest(), bounds.getSouth()],
            [bounds.getEast(), bounds.getSouth()],
            [bounds.getEast(), bounds.getNorth()],
            [bounds.getWest(), bounds.getNorth()],
            [bounds.getWest(), bounds.getSouth()]
        ]]
    };

    // send to Python
    pyObj.receiveGeoJSON(JSON.stringify(geojson));
}

</script>

</body>
</html>
"""


# -------- Python Bridge Object ---------

class WebBridge(QObject):
    geojsonReceived = pyqtSignal(dict)

    @pyqtSlot(str)
    def receiveGeoJSON(self, geojson_str):
        """Receive geoJSON string from JavaScript."""
        geojson = json.loads(geojson_str)
        self.geojsonReceived.emit(geojson)


# -------- Main Window ---------

class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenStreetMap in PyQt + GeoJSON bbox")

        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)

        # Bridge for communication
        self.bridge = WebBridge()

        # Setup JS channel
        self.channel = QWebChannel(self.browser.page())
        self.channel.registerObject("bridge", self.bridge)
        self.browser.page().setWebChannel(self.channel)

        self.browser.setHtml(HTML, QUrl("https://localhost/"))

        # Storage for latest GeoJSON
        self.current_geojson = None
        self.bridge.geojsonReceived.connect(self._on_geojson)

    def _on_geojson(self, geojson):
        """Store and print received GeoJSON."""
        self.current_geojson = geojson
        print("Received GeoJSON:", json.dumps(geojson, indent=2))

    def get_visible_geojson(self):
        """
        Ask JS for visible map area.
        Result arrives asynchronously via WebChannel → _on_geojson().
        """
        js = "getVisibleGeoJSON();"
        self.browser.page().runJavaScript(js)


def main():
    app = QApplication(sys.argv)
    win = MapWindow()
    win.resize(900, 600)
    win.show()

    # Example: request GeoJSON after 2 seconds
    from PyQt6.QtCore import QTimer
    QTimer.singleShot(2000, win.get_visible_geojson)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
