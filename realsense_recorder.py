"""
RealSense-Aufnahme für die MedScribe-QTM-Bridge.

Konfiguriert Depth- und Color-Stream wie die Aufnahme-Zelle in test.ipynb und schreibt
die Rohdaten als .db3 (rosbag). Nach dem Stop liegt die Datei als
<out_dir>/<name>/<name>.db3 und passt damit direkt in die Pipeline (convert() usw.).
<name> enthält die Recording-UUID aus der App. Daneben liegt <name>_recording.json mit den
App-Metadaten und den Aufnahme-Einstellungen (nicht _metadata.json, das schreibt convert()).

Nur pyrealsense2 wird importiert, damit die Bridge ohne die volle Analyse-Umgebung läuft.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pyrealsense2 as rs

LOG = logging.getLogger("bridge.realsense")

# Auflösung/fps-Paare, die für Color und Depth funktionieren (Kopie aus Util/util.py,
# dort nicht importiert, weil util.py rtmlib, matplotlib usw. mitzieht)
resolution_presets = {
    "848x480@60": {"width": 848, "height": 480, "fps": 60},
    "640x360@90": {"width": 640, "height": 360, "fps": 90},
    "480x270@90": {"width": 480, "height": 270, "fps": 90},
}


# Preset je Kategorie/Condition aus den App-Metadaten (Vergleich ohne Groß-/Kleinschreibung).
# Erst wird (category, condition) gesucht, dann (category, None) für alle Conditions.
# Ohne Treffer gilt das Default-Preset (--realsense-preset).
PRESET_BY_CATEGORY = {
    ("gait", None): "848x480@60",
    ("pull test", None): "640x360@90",
    ("tug", None): "848x480@60",
    ("balance", None): "640x360@90",
}


def choose_preset(meta, default_preset_name):
    """Wählt das Preset anhand von category/condition aus den REC:META-Metadaten."""
    if not meta:
        LOG.info("Keine Metadaten – RealSense-Preset %s (Default)", default_preset_name)
        return default_preset_name

    category = str(meta.get("category") or "").strip().lower()
    condition = str(meta.get("condition") or "").strip().lower()
    for key in ((category, condition), (category, None)):
        if key in PRESET_BY_CATEGORY:
            preset_name = PRESET_BY_CATEGORY[key]
            LOG.info(
                "RealSense-Preset %s für %s / %s",
                preset_name,
                meta.get("category"),
                meta.get("condition"),
            )
            return preset_name

    LOG.warning(
        "Kein RealSense-Preset für Kategorie %r / Condition %r – Default %s",
        meta.get("category"),
        meta.get("condition"),
        default_preset_name,
    )
    return default_preset_name


class RealSenseRecorder:
    """Startet/stoppt eine .db3-Aufnahme. start() und stop() blockieren (pipeline.start
    braucht ~1-2 s), daher aus asyncio per asyncio.to_thread aufrufen."""

    def __init__(self, out_dir, preset_name="848x480@60"):
        self.out_dir = Path(out_dir)
        self.default_preset_name = preset_name
        self.pipeline = None
        self.recording_path = None
        self.frame_count = 0
        self.start_time = None
        self.meta = None          # App-Metadaten (REC:META) der laufenden Aufnahme
        self.active_preset_name = None
        self.started_at = None

    @property
    def recording(self):
        return self.pipeline is not None

    def _on_frame(self, frame):
        # Die Frames werden schon im Recorder-Device auf die Platte geschrieben; der Callback
        # gibt sie nur sofort frei und zählt sie für die Kontrolle beim Stop
        self.frame_count += 1

    def start(self, preset_name=None, meta=None, recording_uuid=None):
        """Gibt True zurück, wenn die Aufnahme läuft. preset_name überschreibt das
        Default-Preset für diese Aufnahme. meta wird beim Stop als JSON neben der .db3
        gespeichert; recording_uuid (bereits geprüft) steht schon im vorläufigen Namen."""
        if self.recording:
            LOG.warning("RealSense START ignoriert – nimmt bereits auf")
            return False

        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Vorläufiger Name: der endgültige (mit Zeitstempel des Stops, wie die QTM-Datei)
        # steht erst beim Stop fest. Die UUID macht Reste nach einem Absturz zuordenbar.
        temporary_name = f"_laufend_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        if recording_uuid:
            temporary_name += f"_{recording_uuid}"
        self.recording_path = self.out_dir / f"{temporary_name}.db3"

        preset_name = preset_name or self.default_preset_name
        preset = resolution_presets[preset_name]
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            stream_type=rs.stream.depth,
            width=preset["width"],
            height=preset["height"],
            format=rs.format.z16,
            framerate=preset["fps"],
        )
        config.enable_stream(
            stream_type=rs.stream.color,
            width=preset["width"],
            height=preset["height"],
            format=rs.format.bgr8,
            framerate=preset["fps"],
        )
        config.enable_record_to_file(file_name=str(self.recording_path))

        if not config.can_resolve(pipeline):
            LOG.error("RealSense-Konfiguration %s nicht möglich (Kamera angeschlossen?)", preset)
            return False

        # vor dem Start zurücksetzen: der Callback zählt ab dem ersten Frame
        self.frame_count = 0
        try:
            pipeline.start(config, self._on_frame)
        except RuntimeError as error:
            LOG.error("RealSense-Start fehlgeschlagen: %s", error)
            return False

        self.pipeline = pipeline
        self.start_time = time.monotonic()
        self.meta = meta
        self.active_preset_name = preset_name
        self.started_at = datetime.now().isoformat(timespec="seconds")
        LOG.info(
            "RealSense-Aufnahme gestartet (%dx%d@%d) -> %s",
            preset["width"],
            preset["height"],
            preset["fps"],
            self.recording_path.name,
        )
        return True

    def stop(self, basename, qtm_file=None):
        """Stoppt, benennt die Datei in <out_dir>/<basename>/<basename>.db3 um und schreibt
        <basename>_recording.json mit den Metadaten daneben. Gibt den Pfad der .db3 zurück
        oder None, wenn nicht aufgenommen wurde."""
        if not self.recording:
            LOG.warning("RealSense STOP ignoriert – nimmt gerade nicht auf")
            return None

        pipeline, self.pipeline = self.pipeline, None
        try:
            pipeline.stop()  # schließt auch die .db3
        except RuntimeError as error:
            LOG.error("RealSense-Stop fehlgeschlagen: %s", error)

        duration = time.monotonic() - self.start_time
        measured_fps = self.frame_count / duration if duration > 0 else 0
        LOG.info(
            "RealSense-Aufnahme gestoppt: %d Framesets in %.1f s (%.1f fps)",
            self.frame_count,
            duration,
            measured_fps,
        )

        target_dir = self.out_dir / basename
        target_path = target_dir / f"{basename}.db3"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            self.recording_path.rename(target_path)
        except OSError as error:
            LOG.error(
                "%s konnte nicht nach %s umbenannt werden: %s",
                self.recording_path,
                target_path,
                error,
            )
            target_path = self.recording_path
        else:
            LOG.info("RealSense gespeichert: %s", target_path)

        self._write_sidecar(target_path, qtm_file, duration, measured_fps)
        return target_path

    def _write_sidecar(self, db3_path, qtm_file, duration, measured_fps):
        preset = resolution_presets[self.active_preset_name]
        doc = {
            "db3_file": db3_path.name,
            "qtm_file": qtm_file,
            "outcome": "pending",
            "started_at": self.started_at,
            "stopped_at": datetime.now().isoformat(timespec="seconds"),
            "preset": self.active_preset_name,
            "width": preset["width"],
            "height": preset["height"],
            "fps": preset["fps"],
            "frame_count": self.frame_count,
            "duration_s": round(duration, 3),
            "measured_fps": round(measured_fps, 3),
            "meta": self.meta,
        }
        sidecar_path = db3_path.with_name(db3_path.stem + "_recording.json")
        try:
            sidecar_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as error:
            LOG.error("RealSense-Metadaten %s nicht geschrieben: %s", sidecar_path, error)
            return
        LOG.info("RealSense-Metadaten: %s", sidecar_path.name)
