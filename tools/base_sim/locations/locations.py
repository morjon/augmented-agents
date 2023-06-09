class Location:
    """A single location in the simulated environment."""

    def __init__(self, name, description):
        self.name = name
        """The name of the location."""
        self.description = description
        """The description of the location."""

    def __str__(self):
        return self.name

    def describe(self):
        """Prints the description of the location."""
        print(self.description)


class Locations:
    """
    A collection of locations in the simulated environment stored in a dictionary.
    The keys are the names of the locations and the values are Location objects.
    """

    def __init__(self):
        self.locations = {}

    def add_location(self, name, description):
        """Adds a new location to the collection."""
        self.locations[name] = Location(name, description)

    def get_location(self, name):
        """Returns the Location object with the given name."""
        return self.locations.get(name)

    def __str__(self):
        """Returns a string representation of the collection of locations."""
        return "\n".join([str(location) for location in self.locations.values()])
