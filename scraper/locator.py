from shapely import Polygon
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

class Locator():
    def __init__(self, user_agent : str = "CRoadA_application/1.0.0"):
        self._geolocator = Nominatim(user_agent=user_agent)

        self.geocode = RateLimiter(self._geolocator.geocode, min_delay_seconds=1.5)
        self.reverse = RateLimiter(self._geolocator.reverse, min_delay_seconds=1.5)


    def get_city_name(self, city : str | Polygon):
        if isinstance(city, Polygon):
            center_point = city.centroid

            location = self._geolocator.reverse((center_point.y, center_point.x))

            city_name = location.raw["address"].get("city", None)
            # if city_name is None:
            #     city_name = location.raw["address"].get("town", None)
            return city_name
        elif isinstance(city, str):
            city_name = city.split(",")[0]
            return city_name
        
        return None
    
    def get_city_coords(self, city: str):
            location = self._geolocator.geocode(city, timeout=10)

            if location:
                return (location.latitude, location.longitude)
            return None