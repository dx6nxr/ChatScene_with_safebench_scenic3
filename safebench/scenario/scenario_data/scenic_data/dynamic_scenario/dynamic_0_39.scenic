'''The ego vehicle is driving on a straight road when a pedestrian suddenly crosses from the right front and suddenly stops as the ego vehicle approaches.'''
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until self in ego.lane or self in ego.lane.successor
    while (distance from ego to self) > globalParameters.OPT_STOP_DISTANCE:
        wait
    take SetWalkingSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
# END BEHAVIOR
# BEGIN GEOMETRY
lane = Uniform(*network.lanes)
EgoTrajectory = lane.centerline
EgoSpawnPt = new OrientedPoint on lane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
param OPT_GEO_X_DISTANCE = Range(2, 8)
param OPT_GEO_Y_DISTANCE = Range(15, 50)

IntSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
pedestrian = new Pedestrian right of IntSpawnPt by globalParameters.OPT_GEO_X_DISTANCE,
    with heading IntSpawnPt.heading - 90 deg,  # Heading perpendicular to the road, adjusted for crossing
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to pedestrian > 0
require eventually pedestrian in network.crossingRegion
require eventually ego.canSee(pedestrian, occludingObjects=tuple([])) # Make sure to replace the empty list with a list of all other simulation agents
# END REQUIREMENTS