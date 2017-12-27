from . import table
import pymunk
import math


class Simulation:
    """
    Has:
        A TableInfo
        A State
    """

    def __init__(self, table_info: table.TableInfo):
        self.table_info = table_info
        self.state = self.table_info.get_init_state()

        self.space: pymunk.Space
        self.rod_bodies: [pymunk.Body]
        self.ball_body: pymunk.Body
        self.goal_bodies: (pymunk.Body, pymunk.Body)

        self.space, bodies = self.table_info.get_space()
        self.rod_bodies = bodies["rods"]
        self.ball_body = bodies["ball"]
        self.goal_bodies = tuple(bodies["goals"])

        self.rod_body_idx_cache = _inverse_list(self.rod_bodies, key=id)
        self._hook_velocities()

    # noinspection PyUnusedLocal
    def _get_rod_velocity(self, body: pymunk.Body, gravity, damping, dt):
        rod_idx = self.rod_body_idx_cache[id(body)]
        rod = self.state.rods[rod_idx]
        offset, offset_vel = rod[0]
        angle, angle_vel = rod[1]

        next_foo_x = self.table_info.get_rod_x(rod_idx, angle + angle_vel)
        vel_x = next_foo_x - body.position[0]

        vel_y = offset_vel

        # return vel_x, vel_y
        body.velocity = (vel_x, vel_y)

    def _hook_velocities(self):
        for rod in self.rod_bodies:
            rod.velocity_func = self._get_rod_velocity

    def _fetch_state(self):
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

    def apply_inputs(self, side, input):
        self.state.apply_inputs(self._input_to_absolute(side, input))

    def _input_to_absolute(self, side, input):
        rod_idx, offset_vel, angle_vel = input

        # Side 1's rods are ordered in reverse
        if side == 0:
            ordered = self.table_info.rods
        else:
            ordered = reversed(self.table_info.rods)

        # Find the rod_idx-th rod in the player's order
        # The player's rods are those with side == rod[0]
        idx_left = rod_idx
        for abs_idx, rod in enumerate(ordered):
            if side == rod[0]:
                if idx_left == 0:
                    return abs_idx, offset_vel, angle_vel
                idx_left -= 1

        raise ValueError("rod_idx {} too large".format(rod_idx))

    def tick(self, time):
        self.space.step(time)
        self._fetch_state()

    def get_current_reward(self, player: int):
        """
        :param player: for what player to compute the score;
                       must be:
                        0 for player with the goal at x = 0 coordinate
                        1 for player with the goal at x = MAX_X coordinate)
        :return: a score for current state for player player
        """
        # to be deleted after integration
        assert isinstance(player, int), "Player must be a integer"
        half_goal = self.table_info.goal_width / 2
        player_starting_point = player * self.table_info.length  # i.e X coordinate for goal
        # check for goal
        if player_starting_point - (((-1) ** (1 ^ player)) * self.state.ball.real) <= 0 \
           and ((0.5 - half_goal) < self.state.ball.imag < (0.5 + half_goal)):
            return 100
        # return a score that is positive if the ball is in the opponent half or negative otherwise
        # this score will be weighted by how close is ball to the goal
        sign = -1 if abs(player_starting_point - self.state.ball.real) < (self.table_info.length / 2) else 1
        return sign * abs(self.table_info.length / 2 - self.state.ball.real)


def _inverse_list(items, key):
    results = {}
    for idx, item in enumerate(items):
        results[key(item)] = idx

    return results
