FEATURE_REGISTRY = {

    "auth_failure": {
        "required": ["Time", "UserID", "EventID"],
        "builder": "build_auth_failure",
        "attacks": ["bruteforce", "credential_stuffing"]
    },

    "lateral_movement": {
        "required": ["Time", "UserID", "LogHost"],
        "builder": "build_lateral",
        "attacks": ["lateral_movement", "pass_the_hash"]
    },

    "temporal_anomaly": {
        "required": ["Time", "UserID"],
        "builder": "build_temporal",
        "attacks": ["out_of_hours"]
    },

    "volume_anomaly": {
        "required": ["Time", "UserID"],
        "builder": "build_volume",
        "attacks": ["volume_spike"]
    },

    "rdp_anomaly": {
        "required": ["EventID", "UserID"],
        "builder": "build_rdp",
        "attacks": ["rdp_anomaly"]
    },

    "golden_ticket": {
        "required": ["EventID"],
        "builder": "build_golden_ticket",
        "attacks": ["golden_ticket"]
    },

    "process_anomaly": {
        "required": ["ProcessName", "UserID"],
        "builder": "build_process",
        "attacks": ["service_abuse"]
    },

    "privilege_anomaly": {
        "required": ["UserID", "ProcessName"],
        "builder": "build_privilege",
        "attacks": ["unusual_admin"]
    }
}
