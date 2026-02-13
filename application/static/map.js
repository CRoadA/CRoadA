var pyObj = null;

// WebChannel connection
new QWebChannel(qt.webChannelTransport, function (channel) {
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


// returns GeoJSON bbox of visible area
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

    pyObj.receiveGeoJSON(JSON.stringify(geojson));
}


var drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

var drawControl = new L.Control.Draw({
    draw: {
        polyline: false,
        polygon: false,
        circle: false,
        marker: false,
        circlemarker: false,
        rectangle: {
            shapeOptions: {
                color: '#3388ff',
                weight: 2
            },
            tooltip: {
                start: 'Kliknij i przeciągnij, aby zaznaczyć obszar.'
            }
        }
    },
    edit: {
        featureGroup: drawnItems,
        remove: false,
        edit: false
    }
});
map.addControl(drawControl);

map.on(L.Draw.Event.CREATED, function (e) {
    var layer = e.layer;
    drawnItems.clearLayers();
    drawnItems.addLayer(layer);
    var container = document.createElement('div');
    container.style.textAlign = "center";

    var infoText = document.createElement('p');
    infoText.innerHTML = "<b>Obszar zaznaczony!</b><br>Co chcesz zrobić?";

    // przycisk wysyłania
    var sendBtn = document.createElement('button');
    sendBtn.innerHTML = "Wyślij do backendu";
    sendBtn.style.margin = "5px";
    sendBtn.onclick = function () {
        sendSelectedArea(layer);
        map.closePopup();
    };

    // przycisk anulowania
    var cancelBtn = document.createElement('button');
    cancelBtn.innerHTML = "Usuń";
    cancelBtn.style.margin = "5px";
    cancelBtn.onclick = function () {
        drawnItems.removeLayer(layer);
    };

    container.appendChild(infoText);
    container.appendChild(sendBtn);
    container.appendChild(cancelBtn);

    layer.bindPopup(container).openPopup();
});


function sendSelectedArea(layer) {
    if (!pyObj) {
        console.error("Brak połączenia z Pythonem (pyObj is null)");
        return;
    }
    let bounds = layer.getBounds();
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

    pyObj.receiveGeoJSON(JSON.stringify(geojson));
    drawnItems.removeLayer(layer);
    console.log("Wysłano obszar: ", geojson);
}