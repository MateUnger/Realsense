"""
MedScribe -> Qualisys QTM Bridge

Liest die Serial-Ausgabe des ESP32 (USB) und startet/stoppt bei
"REC:START" / "REC:STOP" eine Aufnahme in QTM über das RT-Protokoll.

Zusätzlich (ab Firmware mit Metadaten-Protokoll):
    REC:META {json}              -> Metadaten der nächsten Aufnahme (Recording-UUID,
                                    Session, Kategorie, Condition, Proband)
    REC:SAVED <recording_uuid>   -> in der App gespeichert
    REC:DISCARDED <recording_uuid> -> in der App verworfen

Die QTM-Datei heißt dann <prefix>_<Zeitstempel>_<recording_uuid>.qtm. Mit --data-dir
(Data-Ordner des QTM-Projekts, vom Bridge-PC aus erreichbar) legt die Bridge daneben
eine gleichnamige .json mit den Metadaten ab und verschiebt verworfene Aufnahmen in
einen Unterordner (bzw. löscht sie mit --delete-discarded).

    iPhone --BLE--> ESP32 --USB-Serial--> diese Bridge --TCP 22223--> QTM

Läuft auf dem PC, an dem der ESP32 per USB hängt (typischerweise der QTM-PC).
QTM 2021.1 unterstützt RT-Protokoll bis 1.22, daher --rt-version 1.22.

Beispiele:
    python medscribe_qtm_bridge.py --list-ports
    python medscribe_qtm_bridge.py --serial COM5
    python medscribe_qtm_bridge.py --serial COM5 --qtm-host 192.168.1.20 --password geheim
    python medscribe_qtm_bridge.py --serial COM5 --data-dir "C:\\QTM Projects\\MedScribe\\Data"
    python medscribe_qtm_bridge.py --manual          # Test ohne ESP32: start/stop/save/discard tippen
"""

import argparse
import asyncio
import json
import logging
import re
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import qtm_rt
from qtm_rt import QRTCommandException, QRTEvent

LOG = logging.getLogger("bridge")

LINE_START = "REC:START"
LINE_STOP = "REC:STOP"
PREFIX_META = "REC:META "
PREFIX_SAVED = "REC:SAVED "
PREFIX_DISCARDED = "REC:DISCARDED "

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class QtmController:
    """Hält die RT-Verbindung zu QTM inkl. Kontrolle und steuert Start/Stop/Save."""

    def __init__(self, args, realsense=None):
        self.args = args
        self.realsense = realsense  # RealSenseRecorder oder None (--realsense-dir "")
        self.conn = None
        self.capturing = False
        self.lock = asyncio.Lock()
        self._waiters = {}
        self._unreachable_logged = False

    # --- Verbindung ---

    def _on_event(self, event):
        LOG.debug("QTM-Event: %s", event.name)
        if event == QRTEvent.EventCaptureStarted:
            self.capturing = True
        elif event == QRTEvent.EventCaptureStopped:
            self.capturing = False
        elif event == QRTEvent.EventWaitingForTrigger:
            LOG.warning("QTM wartet auf externen Trigger (Start on external trigger aktiv?)")
        waiter = self._waiters.get(event)
        if waiter is not None:
            waiter.set()

    def _on_disconnect(self, reason):
        if self.conn is not None:
            LOG.warning("Verbindung zu QTM verloren (%s)", reason)
        self.conn = None

    @property
    def connected(self):
        return self.conn is not None and self.conn.has_transport()

    async def ensure_connected(self):
        if self.connected:
            return True

        a = self.args
        conn = await qtm_rt.connect(
            a.qtm_host,
            port=a.qtm_port,
            version=a.rt_version,
            on_event=self._on_event,
            on_disconnect=self._on_disconnect,
            timeout=a.timeout,
        )
        if conn is None:
            if not self._unreachable_logged:
                LOG.error("QTM unter %s:%d nicht erreichbar – versuche es weiter im Hintergrund",
                          a.qtm_host, a.qtm_port)
                self._unreachable_logged = True
            return False
        self._unreachable_logged = False

        try:
            version = await conn.qtm_version()
            LOG.info("Verbunden mit %s", version.decode(errors="replace"))
            await conn.take_control(a.password)
            LOG.info("Kontrolle über QTM übernommen")

            if a.capture_time > 0:
                xml = (
                    "<QTM_Settings><General>"
                    f"<Capture_Time>{a.capture_time:.1f}</Capture_Time>"
                    "</General></QTM_Settings>"
                )
                try:
                    await conn.send_xml(xml)
                    LOG.info("Max. Capture-Dauer in QTM auf %.0f s gesetzt", a.capture_time)
                except QRTCommandException as e:
                    LOG.warning("Capture-Dauer konnte nicht gesetzt werden (%s) – "
                                "bitte in QTM manuell lang genug einstellen", e)

            # Neue Messung öffnen -> Kameras im Preview, Start geht dann schneller
            await conn.new()
        except (QRTCommandException, asyncio.TimeoutError) as e:
            LOG.error("QTM-Setup fehlgeschlagen: %s", e)
            LOG.error("Ist in QTM 'Allow client control' aktiv und das Passwort korrekt?")
            conn.disconnect()
            return False

        self.conn = conn
        self.capturing = False
        return True

    async def keep_connected(self, interval=5.0):
        """Hintergrund-Task: Verbindung halten, damit Start ohne Verzögerung klappt."""
        while True:
            if not self.connected:
                async with self.lock:
                    await self.ensure_connected()
            await asyncio.sleep(interval)

    async def shutdown(self):
        # Laufende RealSense-Aufnahme beenden, sonst bleibt die .db3 unvollständig
        if self.realsense is not None and self.realsense.recording:
            await asyncio.to_thread(self.realsense.stop,
                                    f"{self.args.prefix}_{datetime.now():%Y-%m-%d_%H-%M-%S}_abgebrochen")
        if self.connected:
            try:
                await self.conn.release_control()
            except Exception:
                pass
            conn, self.conn = self.conn, None
            conn.disconnect()

    # --- Kommandos ---

    async def _send_and_wait(self, command, event, timeout):
        waiter = asyncio.Event()
        self._waiters[event] = waiter
        try:
            await command()
            await asyncio.wait_for(waiter.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            self._waiters.pop(event, None)

    async def start(self, meta=None):
        """Gibt True zurück, wenn QTM die Aufnahme gestartet hat. meta (aus REC:META)
        bestimmt das RealSense-Preset."""
        async with self.lock:
            if not await self.ensure_connected():
                LOG.error("START verworfen – keine Verbindung zu QTM")
                return False
            if self.capturing:
                LOG.warning("START ignoriert – QTM nimmt bereits auf")
                return False
            conn = self.conn
            try:
                await conn.new()
                t0 = time.monotonic()
                ok = await self._send_and_wait(conn.start, QRTEvent.EventCaptureStarted, 15)
                if ok:
                    LOG.info("QTM-Aufnahme gestartet (%.0f ms)", (time.monotonic() - t0) * 1000)
                    if self.realsense is not None:
                        from realsense_recorder import choose_preset
                        preset_name = choose_preset(meta, self.args.realsense_preset)
                        # blockiert ~1-2 s (Kamera-Start), daher im Thread
                        await asyncio.to_thread(
                            self.realsense.start, preset_name, meta,
                            safe_uuid((meta or {}).get("recording_uuid")))
                else:
                    LOG.warning("Start gesendet, aber kein 'Capture started' von QTM erhalten")
                # Auch ohne Bestätigungs-Event: der Start-Befehl ist durch, die
                # Aufnahme gehört ab jetzt zu den Metadaten dieses Starts.
                return True
            except (QRTCommandException, asyncio.TimeoutError) as e:
                LOG.error("Start fehlgeschlagen: %s", e)
                return False

    async def stop(self, basename):
        """Stoppt und speichert als <basename>.qtm. Gibt den Dateinamen zurück, wenn
        QTM das Speichern bestätigt hat, sonst None."""
        async with self.lock:
            if not self.connected:
                LOG.error("STOP verworfen – keine Verbindung zu QTM")
                return None
            if not self.capturing:
                LOG.warning("STOP ignoriert – QTM nimmt gerade nicht auf")
                return None
            conn = self.conn
            try:
                ok = await self._send_and_wait(conn.stop, QRTEvent.EventCaptureStopped, 30)
                if not ok:
                    LOG.warning("Stop gesendet, aber kein 'Capture stopped' von QTM erhalten")
                LOG.info("QTM-Aufnahme gestoppt")

                if self.realsense is not None and self.realsense.recording:
                    await asyncio.to_thread(self.realsense.stop, basename, f"{basename}.qtm")

                # Ohne Pfad speichert QTM im Data-Ordner des aktuellen Projekts
                filename = f"{basename}.qtm"
                resp = await conn.save(filename)
                LOG.info("Gespeichert: %s (%s)", filename, resp.decode(errors="replace"))

                # Messung schließen und direkt neue öffnen -> bereit für den nächsten Start
                await conn.close()
                await conn.new()
                return filename
            except (QRTCommandException, asyncio.TimeoutError) as e:
                LOG.error("Stop/Speichern fehlgeschlagen: %s – bitte in QTM prüfen!", e)
                return None


# --- Metadaten & Speichern/Verwerfen ---

def safe_uuid(value):
    """Nur echte UUIDs landen in Dateinamen (der Wert kommt über BLE von außen)."""
    return value.lower() if isinstance(value, str) and UUID_RE.match(value) else None


class RecordingTracker:
    """Ordnet QTM-Aufnahmen den App-Aufnahmen zu (über die Recording-UUID) und setzt
    die Entscheidung Speichern/Verwerfen aus der App im Data-Ordner um."""

    def __init__(self, args):
        self.args = args
        self.data_dir = Path(args.data_dir) if args.data_dir else None
        self.pending_meta = None   # aus REC:META, wartet auf REC:START
        self.active_meta = None    # gehört zur laufenden QTM-Aufnahme
        self.files = {}            # recording_uuid -> gespeicherter .qtm-Dateiname
        if self.data_dir and not self.data_dir.is_dir():
            LOG.error("--data-dir %s existiert nicht – Metadaten-Dateien und Verwerfen sind "
                      "deaktiviert", self.data_dir)
            self.data_dir = None

    # Eingänge vom ESP32

    def on_meta(self, payload):
        try:
            meta = json.loads(payload)
        except json.JSONDecodeError as e:
            LOG.error("REC:META nicht lesbar (%s) – ignoriert", e)
            return
        if not isinstance(meta, dict) or not safe_uuid(meta.get("recording_uuid")):
            LOG.error("REC:META ohne gültige recording_uuid – ignoriert")
            return
        self.pending_meta = meta
        LOG.info("Metadaten: Recording %s, Session %s, %s / %s, Proband %s",
                 meta.get("recording_uuid"), meta.get("session_uuid") or meta.get("session_id"),
                 meta.get("category"), meta.get("condition"),
                 (meta.get("proband") or {}).get("id"))

    def take_pending_for_start(self):
        """Metadaten des aktuellen START. Werden bei jedem START verbraucht, damit eine
        Aufnahme ohne frische Metadaten (alte App/Firmware) keine veralteten erbt."""
        meta, self.pending_meta = self.pending_meta, None
        return meta

    def started(self, meta):
        self.active_meta = meta
        if meta is None:
            LOG.warning("Aufnahme ohne Metadaten gestartet (App/Firmware ohne Metadaten-Protokoll?)")

    def basename_for_stop(self):
        stamp = f"{self.args.prefix}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        rec = safe_uuid((self.active_meta or {}).get("recording_uuid"))
        return f"{stamp}_{rec}" if rec else stamp

    def stopped(self, filename):
        meta, self.active_meta = self.active_meta, None
        if filename is None or meta is None:
            return
        rec = safe_uuid(meta.get("recording_uuid"))
        self.files[rec] = filename
        self._write_sidecar(filename, meta, outcome="pending")

    def on_outcome(self, rec_uuid, saved):
        rec = safe_uuid(rec_uuid)
        if rec is None:
            LOG.error("Entscheidung mit ungültiger UUID %r – ignoriert", rec_uuid)
            return
        LOG.info("App: Aufnahme %s %s", rec, "GESPEICHERT" if saved else "VERWORFEN")

        filename = self.files.get(rec) or self._find_in_data_dir(rec)
        if filename is None:
            LOG.warning("Keine QTM-Datei zu Recording %s bekannt – Entscheidung kann nicht "
                        "umgesetzt werden", rec)
            return
        if self.data_dir is None:
            if not saved:
                LOG.warning("%s bitte manuell löschen (ohne --data-dir kann die Bridge die Datei "
                            "nicht erreichen)", filename)
            return

        if saved:
            self._update_sidecar(filename, outcome="saved")
            return

        self._update_sidecar(filename, outcome="discarded")
        self._discard(filename)
        self.files.pop(rec, None)

    # Dateien im QTM-Data-Ordner

    def _sidecar_path(self, filename):
        return self.data_dir / (Path(filename).stem + ".json")

    def _write_sidecar(self, filename, meta, outcome):
        if self.data_dir is None:
            return
        if not (self.data_dir / filename).exists():
            LOG.warning("%s nicht in %s gefunden – stimmt --data-dir mit dem Data-Ordner des "
                        "QTM-Projekts überein?", filename, self.data_dir)
        doc = {"qtm_file": filename, "outcome": outcome,
               "stopped_at": datetime.now().isoformat(timespec="seconds"), "meta": meta}
        try:
            self._sidecar_path(filename).write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            LOG.error("Metadaten-Datei für %s nicht geschrieben: %s", filename, e)

    def _update_sidecar(self, filename, outcome):
        path = self._sidecar_path(filename)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {"qtm_file": filename}
        doc["outcome"] = outcome
        doc["decided_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            LOG.error("Metadaten-Datei für %s nicht aktualisiert: %s", filename, e)

    def _find_in_data_dir(self, rec):
        """Fallback nach einem Neustart der Bridge: Datei über die UUID im Namen finden."""
        if self.data_dir is None:
            return None
        matches = sorted(self.data_dir.glob(f"*_{rec}.qtm"))
        return matches[-1].name if matches else None

    def _discard(self, filename):
        files = [self.data_dir / filename, self._sidecar_path(filename)]
        if self.args.delete_discarded:
            for f in files:
                try:
                    f.unlink(missing_ok=True)
                except OSError as e:
                    LOG.error("%s konnte nicht gelöscht werden: %s", f.name, e)
                    return
            LOG.info("Verworfen und gelöscht: %s", filename)
            return

        target = self.data_dir / self.args.discard_dir
        try:
            target.mkdir(exist_ok=True)
            for f in files:
                if f.exists():
                    shutil.move(str(f), str(target / f.name))
        except OSError as e:
            LOG.error("%s konnte nicht verschoben werden (in QTM noch geöffnet?): %s", filename, e)
            return
        LOG.info("Verworfen: %s -> %s", filename, target)


# --- Eingabequellen (laufen in Threads, liefern Zeilen in eine asyncio-Queue) ---

def serial_reader(port, baud, loop, queue):
    import serial

    while True:
        try:
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = baud
            ser.timeout = 1
            # DTR/RTS aus, sonst resettet das Öffnen des Ports den ESP32
            ser.dtr = False
            ser.rts = False
            ser.open()
            LOG.info("Serial %s geöffnet", port)
            while True:
                raw = ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    loop.call_soon_threadsafe(queue.put_nowait, line)
        except serial.SerialException as e:
            LOG.error("Serial %s: %s – neuer Versuch in 2 s", port, e)
            time.sleep(2)


def stdin_reader(loop, queue):
    print("Manueller Modus: 'start', 'stop', 'save' oder 'discard' eingeben (Ctrl+C beendet)")
    last_uuid = None

    def emit(line):
        loop.call_soon_threadsafe(queue.put_nowait, line)

    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd == "start":
            # Simuliert die App: frische Test-Metadaten vor jedem Start
            last_uuid = str(uuid.uuid4()).upper()
            emit(PREFIX_META + json.dumps({
                "v": 1, "recording_uuid": last_uuid, "session_id": 0,
                "session_uuid": None, "category": "Test", "condition": "manuell",
                "proband": None}))
            emit(LINE_START)
        elif cmd == "stop":
            emit(LINE_STOP)
        elif cmd in ("save", "discard"):
            if last_uuid is None:
                print("Noch keine Aufnahme gestartet")
                continue
            emit((PREFIX_SAVED if cmd == "save" else PREFIX_DISCARDED) + last_uuid)
        elif cmd:
            print("Unbekannt – 'start', 'stop', 'save' oder 'discard'")
    emit(None)  # EOF -> Bridge beenden


async def run(args):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    realsense = None
    if args.realsense_dir:
        # erst hier importieren, damit die Bridge ohne pyrealsense2 weiter läuft
        from realsense_recorder import RealSenseRecorder
        realsense = RealSenseRecorder(args.realsense_dir, args.realsense_preset)
    qtm = QtmController(args, realsense)
    tracker = RecordingTracker(args)

    if args.manual:
        target, targs = stdin_reader, (loop, queue)
    else:
        target, targs = serial_reader, (args.serial, args.baud, loop, queue)
    threading.Thread(target=target, args=targs, daemon=True).start()

    keeper = asyncio.create_task(qtm.keep_connected())
    try:
        while True:
            line = await queue.get()
            if line is None:
                break
            if line == LINE_START:
                LOG.info("ESP32: Aufnahme START")
                meta = tracker.take_pending_for_start()
                if await qtm.start(meta):
                    tracker.started(meta)
            elif line == LINE_STOP:
                LOG.info("ESP32: Aufnahme STOP")
                tracker.stopped(await qtm.stop(tracker.basename_for_stop()))
            elif line.startswith(PREFIX_META):
                tracker.on_meta(line[len(PREFIX_META):])
            elif line.startswith(PREFIX_SAVED):
                tracker.on_outcome(line[len(PREFIX_SAVED):].strip(), saved=True)
            elif line.startswith(PREFIX_DISCARDED):
                tracker.on_outcome(line[len(PREFIX_DISCARDED):].strip(), saved=False)
            elif line:
                LOG.debug("ESP32: %s", line)
    finally:
        keeper.cancel()
        await qtm.shutdown()


def list_ports():
    from serial.tools import list_ports as lp

    ports = list(lp.comports())
    if not ports:
        print("Keine seriellen Ports gefunden")
    for p in ports:
        print(f"{p.device:<12} {p.description}")


def choose_port():
    """Sucht COM-Ports und lässt den Benutzer einen auswählen."""
    from serial.tools import list_ports as lp

    while True:
        ports = sorted(lp.comports(), key=lambda x: x.device)
        print("\nGefundene COM-Ports:")
        for i, p in enumerate(ports, 1):
            print(f"  [{i}] {p.device:<8} {p.description}")
        if not ports:
            print("  (keine gefunden)")
        print("  [r] neu suchen   [q] beenden")
        try:
            ans = input("Auswahl (Nummer oder COM-Name): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if ans.lower() == "q":
            return None
        if ans.lower() in ("r", ""):
            if ans == "" and len(ports) == 1:
                return ports[0].device
            continue
        if ans.isdigit() and 1 <= int(ans) <= len(ports):
            return ports[int(ans) - 1].device
        if ans.upper().startswith("COM"):
            return ans.upper()
        print("Ungültige Eingabe")


def main():
    p = argparse.ArgumentParser(description="MedScribe ESP32 -> Qualisys QTM Bridge")
    p.add_argument("--serial", help="Serieller Port des ESP32, z.B. COM5 oder /dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--list-ports", action="store_true", help="Serielle Ports auflisten")
    p.add_argument("--manual", action="store_true", help="Start/Stop per Tastatur (Test ohne ESP32)")
    p.add_argument("--qtm-host", default="127.0.0.1", help="Rechner mit QTM (Default: lokal)")
    p.add_argument("--qtm-port", type=int, default=22223)
    p.add_argument("--rt-version", default="1.22", help="RT-Protokollversion (QTM 2021.1: max 1.22)")
    p.add_argument("--password", default="DSGZ", help="Passwort aus QTM 'Allow client control' (Default: DSGZ)")
    p.add_argument("--prefix", default="MedScribe", help="Präfix für Dateinamen der Aufnahmen")
    p.add_argument("--data-dir",
                   help="Data-Ordner des QTM-Projekts (vom Bridge-PC erreichbar). Aktiviert "
                        ".json-Metadaten neben jeder Aufnahme und das Aussortieren verworfener")
    p.add_argument("--discard-dir", default="Verworfen",
                   help="Unterordner in --data-dir für verworfene Aufnahmen (Default: Verworfen)")
    p.add_argument("--delete-discarded", action="store_true",
                   help="Verworfene Aufnahmen löschen statt verschieben")
    p.add_argument("--capture-time", type=float, default=3600,
                   help="Max. Capture-Dauer in s, die in QTM gesetzt wird (0 = nicht ändern)")
    p.add_argument("--timeout", type=float, default=60,
                   help="Timeout für QTM-Kommandos in s (Speichern kann bei langen Aufnahmen dauern)")
    p.add_argument("--realsense-dir", default=r"C:\Users\munger\realsense_validation_data",
                   help="Ordner für RealSense-Aufnahmen, Datei: <dir>/<name>/<name>.db3 "
                        "(Default: C:\\Users\\munger\\realsense_validation_data). "
                        "--realsense-dir \"\" schaltet die RealSense-Aufnahme ab")
    p.add_argument("--realsense-preset", default="848x480@60",
                   choices=["848x480@60", "640x360@90", "480x270@90"],
                   help="Auflösung@fps der RealSense, wenn Kategorie/Condition aus der App "
                        "keinen Eintrag in PRESET_BY_CATEGORY haben (Default: 848x480@60)")
    p.add_argument("-v", "--verbose", action="store_true", help="Auch übrige ESP32-Ausgaben loggen")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("qtm_rt").setLevel(logging.CRITICAL)  # Fehler loggt die Bridge selbst

    if args.list_ports:
        list_ports()
        return
    if not args.manual and not args.serial:
        args.serial = choose_port()
        if not args.serial:
            return

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nBeendet")


if __name__ == "__main__":
    main()
