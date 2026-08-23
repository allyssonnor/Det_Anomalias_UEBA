import pandas as pd
import os
import glob
import logging
from core.features.feature_registry import FEATURE_REGISTRY

class DatasetInspector:
    """
    O Auditor do Esquadrão:
    Valida cabeçalhos e traduz a capacidade técnica em cobertura de ataques.
    """
    def __init__(self, config):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.label_filename = self.data_cfg.get("label_filename", "labels.csv")
        
        # Mapeamento Semântico: Traduz Features em Ameaças Detectáveis
        self.attack_mapping = {
            "temporal_anomaly": ["Out of Hours", "Unusual Admin Activity", "Service Abuse"],
            "volume_anomaly": ["Brute Force (Volume)", "Data Exfiltration", "DoS Patterns"],
            "auth_failure": ["Brute Force (Login)", "Password Spraying"],
            "lateral_movement": ["Lateral Movement", "Pass-the-Hash (Move)", "RDP Anomaly"],
            "process_anomaly": ["Malware Execution", "Unusual Admin Tools"],
            "privilege_anomaly": ["Privilege Escalation", "Admin Tool Abuse"],
            "rdp_anomaly": ["Remote Access Abuse", "Lateral Movement (RDP)"],
            "golden_ticket": ["Ticket Forgery", "Pass-the-Hash (Auth)"]
        }

        self.maps = {
            'userid': 'UserID', 'user': 'UserID', 'username': 'UserID',
            'timestamp': 'Time', 'time': 'Time', 'date': 'Time',
            'loghost': 'LogHost', 'pc': 'LogHost', 'host': 'LogHost',
            'eventid': 'EventID', 'id': 'EventID',
            'processname': 'ProcessName', 'filename': 'ProcessName'
        }

    def inspect_all(self):
        """Executa a auditoria e imprime o relatório no terminal."""
        print(f"\n{'='*60}")
        print(f"🔎 INSPEÇÃO TÉCNICA E SEMÂNTICA")
        print(f"{'='*60}")
        
        train_path = self.data_cfg.get("train_path")
        if not train_path or not os.path.exists(train_path):
            print(f"❌ Erro: Caminho de treino não encontrado.")
            return False

        headers = self._get_headers(train_path)
        if not headers: return False

        clean_headers, _ = self._analyze_mapping(headers)
        
        print(f"Dataset: {os.path.basename(train_path)}")
        missing_vital = [c for c in ['UserID', 'Time'] if c not in clean_headers]
        if missing_vital:
            print(f"🚨 STATUS: FALHA (Faltam colunas vitais: {missing_vital})")
            return False
        
        self._report_attack_coverage(clean_headers)
        print(f"{'='*60}\n")
        return True

    def _get_headers(self, path):
        try:
            target = path
            if os.path.isdir(path):
                files = sorted(glob.glob(os.path.join(path, "*.*")))
                files = [f for f in files if self.label_filename not in f]
                if not files: return None
                target = files[0]
            ext = os.path.splitext(target)[1].lower()
            df = pd.read_json(target, lines=True, nrows=1) if "json" in ext else pd.read_csv(target, nrows=1)
            return [str(c).strip() for c in df.columns]
        except: return None

    def _analyze_mapping(self, headers):
        clean = [self.maps.get(h.lower(), h) for h in headers]
        return clean, ""

    def _report_attack_coverage(self, clean_headers):
        available_cols = set(clean_headers)
        active_attacks = set()
        print(f"\n🎯 COBERTURA DE ATAQUES ESTIMADA:")
        print(f"{'-'*60}")
        
        for group, attacks in self.attack_mapping.items():
            spec = FEATURE_REGISTRY.get(group)
            if not spec: continue
            
            if set(spec["required"]).issubset(available_cols):
                for a in attacks: active_attacks.add(a)
                print(f"  ✅ {group.ljust(18)} -> {', '.join(attacks)}")
            else:
                missing = set(spec["required"]) - available_cols
                print(f"  ❌ {group.ljust(18)} -> (Faltam: {missing})")

        print(f"{'-'*60}")
        print(f"🚀 O Esquadrão monitoriza {len(active_attacks)} tipos de ameaças.")
