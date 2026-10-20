import logging
import config

def setup():
    # Ordner erstellen, falls er nicht existiert
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%d-%m-%Y %H:%M:%S")

    # Haupt-Logger für das Projekt holen
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # 1. Datei-Handler (schreibt in output.log)
    file_handler = logging.FileHandler(config.LOGS_DIR / "output.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Konsolen-Handler (gibt im Terminal aus)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Verhindert doppelte Ausgaben durch Weitergabe an den Root-Logger
    logger.propagate = False

    return logger