from . import table, input
import pymunk


class Simulation:
    """
    Has:
        A TableInfo
        A State
    """

    def __init__(self, table_info: table.TableInfo):
        self.table_info = table_info
        self.state = self.table_info.get_init_state()
        self.inputs_applied = []

        self.space : pymunk.Space
        self.rod_bodies : [pymunk.Body]
        self.ball_body : pymunk.Body
        self.goal_bodies : (pymunk.Body, pymunk.Body)

        self.space, bodies = self.table_info.get_space()
        self.rod_bodies = bodies["rods"]
        self.ball_body = bodies["ball"]
        self.goal_bodies = tuple(bodies["goals"])

        self.rod_idx_cache = inverse_list(self.rod_bodies, key=id)
        self.hook_velocities()

    def get_rod_velocity(self, body, gravity, damping, dt):
        rod_idx = self.rod_idx_cache[id(body)]
        rod = self.state.rods[rod_idx]
        offset, offset_vel = rod[0]
        angle, angle_vel = rod[1]

        next_foo_x = self.table_info.get_rod_x(rod_idx, angle + angle_vel)
        vel_x = next_foo_x - body.position[0]

        vel_y = offset_vel

        return (vel_x, vel_y)

    def hook_velocities(self):
        for rod in self.rod_bodies:
            for body in rod:
                body.velocity_func = self.get_rod_velocity

    def fetch_state(self):
        ball_offset = self.ball_body.position
        ball_velocity = self.ball_body.velocity
        self.state.ball = (complex(ball_offset[0], ball_offset[1]),
                           complex(ball_velocity[0], ball_velocity[1]))
        for idx, body in enumerate(self.rod_bodies):
            rod_offset = self.table_info.get_rod_offset(idx, body.position[1])
            rod_angle = self.table_info.get_rod_angle(idx, body.position[0])
            rod_last_angle = self.table_info.get_rod_angle(idx, body.position[0] - body.velocity[0])
            self.state.rods[idx] = ((rod_offset, body.velocity[1]),
                                    (rod_angle, rod_angle - rod_last_angle))

    def apply_inputs(self, input_to_apply:input.Input):
        assert len(self.inputs_applied) < 2 and input_to_apply.side not in self.inputs_applied
        self.inputs_applied.append(input_to_apply.side)

        self.state.apply_inputs(input)

    def tick(self, time):
        # TODO: set velocities using body.velocity_func
        self.space.step(time)


def inverse_list(items, key):
    results = {}
    for idx, item in enumerate(items):
        results[key(item)] = idx

    return results
