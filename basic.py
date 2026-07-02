import time
import mujoco
import mujoco.viewer

# Load model
model = mujoco.MjModel.from_xml_path("basic.xml")

# Runtime state
data = mujoco.MjData(model)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)