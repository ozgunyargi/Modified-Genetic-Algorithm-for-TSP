from haversine import haversine
from dataclasses import dataclass, field

@dataclass
class City:
    name: str
    lon: float = field(repr=False)
    lat: float = field(repr=False)

    def distance (self, city):

        dist = haversine( (self.lat, self.lon),(city.lat,city.lon) )
        return dist
        
class Fitness:

    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0

    def __str__(self) -> str:
        city_name_list = [city.name for city in self.route]
        output_str = ""
        for city in city_name_list:
            output_str += city+"-"
        return output_str[:-1]

    def calculate_fitness(self):

        if self.fitness == 0:

            for city_index in range(len(self.route)):
                if city_index != len(self.route)-1:
                    dist = self.route[city_index].distance(self.route[city_index+1])
                else:
                    dist = self.route[0].distance(self.route[len(self.route)-1])
                self.distance += dist

            self.fitness = (1/self.distance) # Convert the task to maximization problem

        return self.fitness


def main():
    city = Cities("1")
    print(city.name)

if __name__ == '__main__':
    main()
