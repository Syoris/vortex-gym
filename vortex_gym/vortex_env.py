from pydantic import BaseModel
from abc import ABC
from pyvortex.vortex_interface import VortexInterface, AppMode
from pathlib import Path
from vortex_gym import ASSETS_DIR


class VortexEnv:
    def __init__(
        self,
        h=0.01,
        config_file='config.vxc',
        content_file='Kinova Gen2 Unjamming/Scenes/kinova_peg-in-hole.vxscene',
        inputs_interface=VX_Interface(),
        outputs_interface=VX_Interface(),
        params_interface=VX_Interface(),
    ):
        # self.config = config

        self.sim_time = 0.0
        self.step_count = 0  # Step counter

        self.inputs_interface = inputs_interface
        self.outputs_interface = outputs_interface
        self.params_interface = params_interface

        """Initialize environment (Vortex) parameters"""
        # Vortex
        self.h = h  # Simulation time step
        self.config_file_ = Path(config_file)
        self.content_file_ = Path(content_file)

        """ Load Vortex Scene """
        # Define the setup and scene file paths
        self.setup_file = ASSETS_DIR / self.config_file_  # 'config_withoutgraphics.vxc'
        self.content_file = ASSETS_DIR / self.content_file_

        # Create the Vortex Application
        self.vx = VortexInterface()
        self.vx.create_application(self.setup_file)

        self.vx.load_scene(self.content_file)

        self.h_sim_ = self.vx.get_simulation_time_step()
        if self.h != self.h_sim_:
            raise ValueError(f'Simulation time step mismatch between application and config: {self.h} != {self.h_sim_}')

        self.vx.load_display()
        self.vx.render_display(active=True)

    def step(self):
        """To step the simulation"""
        # self._send_joint_target_vel(self.command)

        self.vx.app.update()

        self.sim_time = self.vx.app.getSimulationTime()
        # self.obs = self._get_robot_state()
