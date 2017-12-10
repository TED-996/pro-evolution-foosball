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
        self.space : pymunk.Space = None
        self.rod_bodies : [pymunk.Body]
        self.space, self.rod_bodies = self.table_info.get_space()

    def apply_inputs(self, input:input.Input):
        assert len(self.inputs_applied) < 2 and input.side not in self.inputs_applied
        self.inputs_applied.append(input.side)

        self.state.apply_inputs(input)

    def tick(self, time):
        # TODO: set velocities using body.velocity_func
        self.space.step(time)
