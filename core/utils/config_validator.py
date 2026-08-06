class ConfigValidator:

    def __init__(self, config):
        self.config = config

    def validate(self):

        cfg = self.config

        # =========================================================
        # DATA
        # =========================================================
        if "data" not in cfg:
            raise ValueError("Missing 'data' section")

        data_cfg = cfg["data"]

        # aceita path único OU split
        if not (
            "path" in data_cfg or
            (
                "train_path" in data_cfg and
                "val_path" in data_cfg and
                "test_path" in data_cfg
            )
        ):
            raise ValueError(
                "data must have either 'path' OR "
                "'train_path', 'val_path', 'test_path'"
            )

        # =========================================================
        # MODEL
        # =========================================================
        if "model" not in cfg:
            raise ValueError("Missing 'model' section")

        if "type" not in cfg["model"]:
            raise ValueError("Missing 'model.type'")

        if "sequence_length" not in cfg["model"]:
            raise ValueError("Missing 'model.sequence_length'")

        # =========================================================
        # SAMPLING
        # =========================================================
        if "sampling" in cfg:

            mode = cfg["sampling"].get("mode", "full")

            if mode not in ["full", "debug"]:
                raise ValueError("sampling.mode must be 'full' or 'debug'")

        print("✅ Config validation passed")