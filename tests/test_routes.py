import unittest

from sim.routes import haversine_km, sea_route


class RouteTests(unittest.TestCase):
    def test_sea_route_uses_searoute_network_not_direct_line(self):
        brevik = (59.05, 9.70)
        oygarden = (60.62, 4.84)

        route = sea_route(brevik, oygarden)
        direct_km = haversine_km(brevik, oygarden)

        self.assertEqual(route.provider, "searoute")
        self.assertGreater(len(route.coordinates), 2)
        self.assertGreater(route.distance_km, direct_km)
        self.assertLess(haversine_km(route.coordinates[0], brevik), 10.0)
        self.assertLess(haversine_km(route.coordinates[-1], oygarden), 35.0)


if __name__ == "__main__":
    unittest.main()
