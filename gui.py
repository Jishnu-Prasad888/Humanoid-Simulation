import json
import os
import customtkinter as ctk
import tkinter.filedialog as fd

CONFIG_PATH = os.getenv("HUMANOID_CONFIG", "config.json")


DEFAULT_CONFIG = {
    "pin_root": True,
    "keyframe_name": "stand_bi",
    "friction": [4.5, 0.1, 0.01],
    "gains": {
        "hip": {"kp": 80.0, "kd": 35.0, "ki": 0.0},
        "knee": {"kp": 80.0, "kd": 35.0, "ki": 0.0},
        "ankle": {"kp": 75.0, "kd": 30.0, "ki": 0.0},
        "abdomen": {"kp": 35.0, "kd": 15.0, "ki": 0.0},
        "shoulder": {"kp": 25.0, "kd": 12.0, "ki": 0.0},
        "elbow": {"kp": 25.0, "kd": 12.0, "ki": 0.0},
    },
    "pose": {
        "root_z": 1.05,
        "abdomen": [0.0, 0.0, 0.0],
        "right_leg": [-0.10, 0.03, 0.04, -0.14, -0.01, 0.02],
        "left_leg": [-0.10, -0.03, -0.04, -0.14, -0.01, -0.02],
        "right_arm": [0.02, -0.18, -0.28],
        "left_arm": [-0.02, -0.18, -0.28],
    },
    "smoothing": 0.4,
    "i_clamp": 0.2,
}


def ensure_config(path: str) -> dict:
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return dict(DEFAULT_CONFIG)
    with open(path, "r") as f:
        cfg = json.load(f)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    def deep_merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
                deep_merge(dst[k], v)
            else:
                dst[k] = v
    deep_merge(merged, cfg)
    return merged


class NumericField(ctk.CTkFrame):
    def __init__(self, master, label: str, value: float, step: float = 0.01, callback=None, width=80):
        super().__init__(master)
        self.step = step
        self.callback = callback
        self.var = ctk.StringVar(value=f"{value:.4f}")
        ctk.CTkLabel(self, text=label, width=90, anchor="w").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.entry = ctk.CTkEntry(self, textvariable=self.var, width=width)
        self.entry.grid(row=0, column=1, padx=2, pady=2)
        up = ctk.CTkButton(self, text="▲", width=26, height=26, command=self.increment)
        down = ctk.CTkButton(self, text="▼", width=26, height=26, command=self.decrement)
        up.grid(row=0, column=2, padx=1)
        down.grid(row=0, column=3, padx=1)
        self.entry.bind("<Return>", self.on_change)
        self.entry.bind("<FocusOut>", self.on_change)
        self.entry.bind("<Up>", lambda e: self.increment())
        self.entry.bind("<Down>", lambda e: self.decrement())
        self.entry.bind("<MouseWheel>", self.on_wheel)

    def get_value(self) -> float:
        try:
            return float(self.var.get())
        except ValueError:
            return 0.0

    def set_value(self, val: float):
        self.var.set(f"{val:.4f}")

    def on_change(self, event=None):
        if self.callback:
            self.callback()

    def increment(self):
        self.set_value(self.get_value() + self.step)
        self.on_change()

    def decrement(self):
        self.set_value(self.get_value() - self.step)
        self.on_change()

    def on_wheel(self, event):
        delta = self.step if event.delta > 0 else -self.step
        self.set_value(self.get_value() + delta)
        self.on_change()


class ConfigUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Humanoid Controller Config")
        self.geometry("820x820")
        self.resizable(True, True)
        self.config_path = CONFIG_PATH
        self.cfg = ensure_config(self.config_path)

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        self.fields = {}

        self.build_general()
        self.build_friction()
        self.build_gains()
        self.build_pose()
        self.build_misc()

        action_row = ctk.CTkFrame(self.scroll)
        action_row.pack(fill="x", pady=8, padx=6)
        self.path_label = ctk.CTkLabel(action_row, text=f"Active: {os.path.abspath(self.config_path)}", anchor="w")
        self.path_label.pack(fill="x", padx=4, pady=4)
        btn_row = ctk.CTkFrame(self.scroll)
        btn_row.pack(fill="x", pady=4, padx=6)
        ctk.CTkButton(btn_row, text="Save Now", command=self.save, width=120).pack(side="left", padx=4, pady=4)
        ctk.CTkButton(btn_row, text="Save As...", command=self.save_as, width=120).pack(side="left", padx=4, pady=4)
        ctk.CTkButton(btn_row, text="Import...", command=self.import_into_current, width=120).pack(side="left", padx=4, pady=4)

    def save(self):
        with open(self.config_path, "w") as f:
            json.dump(self.cfg, f, indent=2)
        self.update_path_label()

    def save_as(self):
        path = fd.asksaveasfilename(defaultextension=".json",
                                    initialfile=os.path.basename(self.config_path),
                                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.config_path = path
            self.save()

    def import_into_current(self):
        path = fd.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "r") as f:
            cfg_in = json.load(f)
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        def deep_merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
                    deep_merge(dst[k], v)
                else:
                    dst[k] = v
        deep_merge(merged, cfg_in)
        self.cfg = merged
        self.refresh_fields()
        self.save()

    def mark_change(self):
        self.save()

    def update_path_label(self):
        self.path_label.configure(text=f"Active: {os.path.abspath(self.config_path)}")

    def build_general(self):
        frame = ctk.CTkFrame(self.scroll)
        frame.pack(fill="x", pady=6)
        ctk.CTkLabel(frame, text="General", font=("Arial", 14, "bold")).pack(anchor="w", padx=6, pady=4)
        self.pin_root_var = ctk.BooleanVar(value=self.cfg.get("pin_root", True))
        pin_switch = ctk.CTkSwitch(frame, text="Pin Root", variable=self.pin_root_var,
                                   command=self.on_pin_root)
        pin_switch.pack(anchor="w", padx=8, pady=2)

    def on_pin_root(self):
        self.cfg["pin_root"] = bool(self.pin_root_var.get())
        self.mark_change()

    def build_friction(self):
        frame = ctk.CTkFrame(self.scroll)
        frame.pack(fill="x", pady=6)
        ctk.CTkLabel(frame, text="Friction [slide, spin, roll]", font=("Arial", 14, "bold")).pack(anchor="w", padx=6, pady=4)
        fr = self.cfg.get("friction", [3.0, 0.1, 0.01])
        self.friction_fields = []
        names = ["slide", "spin", "roll"]
        row = ctk.CTkFrame(frame)
        row.pack(fill="x", padx=6, pady=4)
        for i, label in enumerate(names):
            field = NumericField(row, label, fr[i] if i < len(fr) else 0.0, step=0.05, callback=self.on_friction_change, width=90)
            field.grid(row=0, column=i, padx=4, pady=2)
            self.friction_fields.append(field)

    def on_friction_change(self):
        self.cfg["friction"] = [f.get_value() for f in self.friction_fields]
        self.mark_change()

    def build_gains(self):
        frame = ctk.CTkFrame(self.scroll)
        frame.pack(fill="x", pady=6)
        ctk.CTkLabel(frame, text="Gains", font=("Arial", 14, "bold")).pack(anchor="w", padx=6, pady=4)
        groups = ["hip", "knee", "ankle", "abdomen", "shoulder", "elbow"]
        for g in groups:
            row = ctk.CTkFrame(frame)
            row.pack(fill="x", padx=6, pady=3)
            ctk.CTkLabel(row, text=g.capitalize(), width=80, anchor="w").grid(row=0, column=0, padx=4)
            gains = self.cfg["gains"].get(g, {"kp": 0, "kd": 0, "ki": 0})
            kp_field = NumericField(row, "Kp", gains.get("kp", 0), step=1.0, callback=lambda g=g: self.on_gain_change(g))
            kd_field = NumericField(row, "Kd", gains.get("kd", 0), step=1.0, callback=lambda g=g: self.on_gain_change(g))
            ki_field = NumericField(row, "Ki", gains.get("ki", 0), step=0.5, callback=lambda g=g: self.on_gain_change(g))
            kp_field.grid(row=0, column=1, padx=2)
            kd_field.grid(row=0, column=2, padx=2)
            ki_field.grid(row=0, column=3, padx=2)
            self.fields[("gains", g, "kp")] = kp_field
            self.fields[("gains", g, "kd")] = kd_field
            self.fields[("gains", g, "ki")] = ki_field

    def on_gain_change(self, group):
        self.cfg["gains"][group]["kp"] = self.fields[("gains", group, "kp")].get_value()
        self.cfg["gains"][group]["kd"] = self.fields[("gains", group, "kd")].get_value()
        self.cfg["gains"][group]["ki"] = self.fields[("gains", group, "ki")].get_value()
        self.mark_change()

    def build_pose(self):
        frame = ctk.CTkFrame(self.scroll)
        frame.pack(fill="x", pady=6)
        ctk.CTkLabel(frame, text="Pose (custom stand)", font=("Arial", 14, "bold")).pack(anchor="w", padx=6, pady=4)

        # root z
        rz_row = ctk.CTkFrame(frame)
        rz_row.pack(fill="x", padx=6, pady=2)
        self.root_z_field = NumericField(rz_row, "root_z", self.cfg["pose"].get("root_z", 1.05), step=0.01, callback=self.on_pose_change)
        self.root_z_field.grid(row=0, column=0, padx=2)

        def array_fields(key, labels, step=0.01):
            row = ctk.CTkFrame(frame)
            row.pack(fill="x", padx=6, pady=2)
            ctk.CTkLabel(row, text=key, width=90, anchor="w").grid(row=0, column=0, padx=4)
            arr = self.cfg["pose"].get(key, [0.0] * len(labels))
            for i, lab in enumerate(labels):
                field = NumericField(row, lab, arr[i] if i < len(arr) else 0.0, step=step, callback=lambda k=key: self.on_pose_change())
                field.grid(row=0, column=i + 1, padx=2)
                self.fields[("pose", key, i)] = field

        array_fields("abdomen", ["z", "y", "x"], step=0.01)
        array_fields("right_leg", ["hipx", "hipz", "hipy", "knee", "ank_y", "ank_x"], step=0.01)
        array_fields("left_leg", ["hipx", "hipz", "hipy", "knee", "ank_y", "ank_x"], step=0.01)
        array_fields("right_arm", ["sh1", "sh2", "elb"], step=0.01)
        array_fields("left_arm", ["sh1", "sh2", "elb"], step=0.01)

    def on_pose_change(self):
        pose = self.cfg["pose"]
        pose["root_z"] = self.root_z_field.get_value()
        for key in ["abdomen", "right_leg", "left_leg", "right_arm", "left_arm"]:
            count = 6 if "leg" in key else (3 if "arm" in key or key == "abdomen" else 0)
            pose[key] = [self.fields[("pose", key, i)].get_value() for i in range(count)]
        self.mark_change()

    def build_misc(self):
        frame = ctk.CTkFrame(self.scroll)
        frame.pack(fill="x", pady=6)
        ctk.CTkLabel(frame, text="Misc", font=("Arial", 14, "bold")).pack(anchor="w", padx=6, pady=4)
        misc_row = ctk.CTkFrame(frame)
        misc_row.pack(fill="x", padx=6, pady=4)
        self.smoothing_field = NumericField(misc_row, "smoothing", self.cfg.get("smoothing", 0.4), step=0.05, callback=self.on_misc_change)
        self.i_clamp_field = NumericField(misc_row, "i_clamp", self.cfg.get("i_clamp", 0.2), step=0.05, callback=self.on_misc_change)
        self.smoothing_field.grid(row=0, column=0, padx=4)
        self.i_clamp_field.grid(row=0, column=1, padx=4)

    def on_misc_change(self):
        self.cfg["smoothing"] = self.smoothing_field.get_value()
        self.cfg["i_clamp"] = self.i_clamp_field.get_value()
        self.mark_change()

    def refresh_fields(self):
        # general
        self.pin_root_var.set(self.cfg.get("pin_root", True))

        # friction
        fr = self.cfg.get("friction", [3.0, 0.1, 0.01])
        for i, f in enumerate(self.friction_fields):
            f.set_value(fr[i] if i < len(fr) else 0.0)

        # gains
        for (section, g, key), field in self.fields.items():
            if section != "gains":
                continue
            field.set_value(self.cfg.get("gains", {}).get(g, {}).get(key, 0.0))

        # pose
        self.root_z_field.set_value(self.cfg.get("pose", {}).get("root_z", 1.05))

        def update_pose(key, count):
            arr = self.cfg.get("pose", {}).get(key, [0.0] * count)
            for i in range(count):
                self.fields[("pose", key, i)].set_value(arr[i] if i < len(arr) else 0.0)

        update_pose("abdomen", 3)
        update_pose("right_leg", 6)
        update_pose("left_leg", 6)
        update_pose("right_arm", 3)
        update_pose("left_arm", 3)

        # misc
        self.smoothing_field.set_value(self.cfg.get("smoothing", 0.4))
        self.i_clamp_field.set_value(self.cfg.get("i_clamp", 0.2))

        self.update_path_label()


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = ConfigUI()
    app.mainloop()


if __name__ == "__main__":
    main()
