class PegTask:
    def __init__(self):
        ...
        # # TODO: Move to task parameters
        # self.t_init_step = self.task_cfg.env.t_init_step  # Time to move arm to the insertion position
        # self.t_pause = self.task_cfg.env.t_pause  # Pause time
        # self.t_pre_insert = self.task_cfg.env.t_pre_insert  # used to be 0.8 for 7DOF
        # self.t_insertion = self.task_cfg.env.t_insertion

        # # Steps numbers
        # self.init_steps = int(self.t_init_step / self.h)  # Initialization steps
        # self.pause_steps = int(self.t_pause / self.h)  # Pause time step after one phase
        # self.pre_insert_steps = int(self.t_pre_insert / self.h)  # go up to the insertion phase
        # self.insertion_steps = int(self.t_insertion / self.h)  # Insertion time (steps)
        # self.max_insertion_steps = 2.0 * self.insertion_steps  # Maximum allowed time to insert

        # # Peg and hole parameters
        # self.xpos_hole = self.task_cfg.env.xpos_hole  # x position of the hole
        # self.ypos_hole = self.task_cfg.env.ypos_hole  # y position of the hole
        # (
        #     self.min_misalign,
        #     self.max_misalign,
        # ) = self.task_cfg.env.misalignment_range  # maximum misalignment of joint 7 [deg]

        # # joint 6 misalignment, negative and positive [deg]
        # self.insertion_misalign = np.random.uniform(self.min_misalign, self.max_misalign)
        # # self.insertion_misalign = (np.pi/180.0) * self.max_misalign    # joint 6 misalignment, set max

        # self.pre_insertz = self.task_cfg.env.pre_insertz  # z distance to be covered in pre-insertion phase
        # self.insertz = (
        #     self.task_cfg.env.insertz
        # )  # z distance to be covered in insertion phase, though may change with actions
