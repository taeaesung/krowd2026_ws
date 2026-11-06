# Stable track ID supplied by perception. Reuse the same ID across frames.
int64 id

# Obstacle center and velocity expressed in ObstacleArray.header.frame_id.
geometry_msgs/Point center
geometry_msgs/Vector3 velocity

# Conservative horizontal enclosing-circle radius in metres.
float64 radius

# Perception confidence in [0, 1].
float64 confidence
