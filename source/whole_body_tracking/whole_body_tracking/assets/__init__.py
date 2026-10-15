import os

# Conveniences to other module directories via relative paths
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))
ROBOT_X2_ULTRA_ASSET_DIR = os.path.join(ASSET_DIR, "robots", "x2_ultra_description")
