# polygon-operations
simplifiers polygons and creates valid polygons from non-valid polygons

Basically, two functions are provided in this package:
one function to simplify a polygon and another to create a valid from a non-valid polygon.
They are called simplify_polygon() or create_valid_polygon(), respectively.
Parameters concerning the desired precision should be supplied.

The simplify_polygon() function decreases the number of points used to define the polygon,
which may dramatically reduce the calculation time for many problems.

The create_valid_polygon() function tries to create a valid polygon based on a non-valid one.
This is a non-evident task that cannot, to the very best of my understanding,
be generalized to work on every non-valid polygon as the solution may not be unique.
Again, Parameters concerning the desired precision should be supplied.

