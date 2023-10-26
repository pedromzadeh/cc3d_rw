from cc3d.core.PySteppables import *
import numpy as np


class RandomWalker(SteppableBasePy):
    def __init__(self, polarity_config, dt=0.01, frequency=1):
        """
        Initialize the steppable that models cell polarity as a random walker.

        Parameters
        ----------
        polarity_config : list of dict
            Specifies the polarity module for each cell type (len of config). Keys are
            "v0" (cell velocity magnitude) and "D" (angular diffusion coeff).

        dt : float, optional
            Sets the time delta between each MCS. Ficticious parameter. By default
            0.01.

        frequency : int, optional
            Steppable runs every `frequency * MCS`; by default 10, which runs the
            steppable every 10th Monte Carlo Step.
        """
        super().__init__(frequency)
        self.dt = dt
        self.polarity_config = polarity_config

    def start(self):
        self._verify_polarity_config(self.polarity_config)

        # all non-medium cells
        for cell in self.cell_list:
            cell.dict["v0"] = self.polarity_config[cell.type - 1]["v0"]
            cell.dict["D"] = self.polarity_config[cell.type - 1]["D"]
            cell.dict["cm_record"] = [[cell.xCOM, cell.yCOM]]
            self.init_cell_polarity(cell)

        # init plot windows
        self.com_window = self.add_new_plot_window(
            title="COM Track", x_axis_title="X", y_axis_title="Y", grid=False
        )
        self.com_window.add_plot("Center of Mass", style="dot", size=1)
        self.com_window.add_data_point(
            "Center of Mass",
            [0, 0, self.dim.x, self.dim.x],
            [0, self.dim.y, 0, self.dim.y],
        )

    def step(self, mcs):
        for cell in self.cell_list:
            cell.dict["cm_record"].append([cell.xCOM, cell.yCOM])
            self.update_cell_polarity(cell, D=cell.dict["D"], dt=self.dt)

            # plot center of mass
            self.com_window.add_data_point("Center of Mass", cell.xCOM, cell.yCOM)

    def finish(self):
        import os
        import yaml

        os.makedirs("output", exist_ok=True)
        for cell in self.cell_list:
            with open(f"output/cellID_{cell.id}.json", "w") as f:
                metadata = {
                    "id": cell.id,
                    "type": cell.type,
                    "D": cell.dict["D"],
                    "v0": cell.dict["v0"],
                }
                yaml.dump(
                    {
                        "metadata": metadata,
                        "cm_track": cell.dict["cm_record"],
                    },
                    f,
                )

    def _verify_polarity_config(self, config):
        n_types = np.unique([c.type for c in self.cell_list]).size
        if n_types != len(config):
            raise ValueError(
                f"Have {n_types} cell types, but {len(config)} polarity configs."
            )
        if not all([k in d for k in ["v0", "D"] for d in config]):
            raise KeyError("Make sure all configs have keys 'v0' and 'D'.")

    @staticmethod
    def update_cell_polarity(cell, D, dt):
        """
        Internally updates cell polarity.

        Parameters
        ----------
        cell : cc3d cell instance
            The cell for which to update the polarity.

        D : float
            Angular diffusion coeff.

        dt : float
            Monte Carlo Step time. It's a ficticious parameter that
            modulates the variance of the random number. Could be set
            to 1, just lower D.
        """
        theta = np.arctan2(cell.lambdaVecY, cell.lambdaVecX)
        theta += np.random.normal(0, 1) * np.sqrt(2 * D * dt)
        v = lambda x: cell.dict["v0"] * np.array([np.cos(x), np.sin(x)])
        cell.lambdaVecX, cell.lambdaVecY = v(theta)

    @staticmethod
    def init_cell_polarity(cell):
        """
        Internally sets cell polarity to a random direction.

        Parameters
        ----------
        cell : cc3d cell instance
            The cell for which to set the polarity.
        """
        v = lambda x: cell.dict["v0"] * np.array([np.cos(x), np.sin(x)])
        cell.lambdaVecX, cell.lambdaVecY = v(np.random.uniform(0, 2 * np.pi))
