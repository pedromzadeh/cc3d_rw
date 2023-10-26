from cc3d import CompuCellSetup
from RandomWalkingCellSteppables import RandomWalker

CompuCellSetup.register_steppable(
    steppable=RandomWalker(
        polarity_config=[
            {"v0": 30, "D": 2},
        ]
    )
)

CompuCellSetup.run()
