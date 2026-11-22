# CaracalStreamer
# This is an object which takes a Caracal Data directory and creates AudioPacket instances from each session in the directory. 
# This code is built to interface caracal-python with the rest of this programme
# CaracalStreamer inherits behaviour from the PipelineLink object
# To init a CaracalStreamer, you just need to pass a top level directory where sub-directories contain data from caracal sessions. The logic inside will visit each session's log file and create an AudioPacket instance for each, these will be called back to whatever callback you registered. 
# Note, due to inheritance, it is technically possible to pipe AudioPackets into CaracalStreamer, but this implementation doesnt make use of that feature. 

import os
from pathlib import Path
from hashlib import sha256

from caracal import SyslogParser

from src.PipelineLink import PipelineLink
from src.AudioPacket import AudioPacket
from src.AudioPacketSource import AudioPacketSource

class CaracalStreamer(PipelineLink, AudioPacketSource):
    """Pipeline source that discovers CARACAL session syslog files and emits AudioPackets.

    The streamer finds syslog.txt files under a root directory and converts each
    session into an AudioPacket which is emitted downstream via next_audio.
    """

    def __init__(self, rootpath: str):
        """Create a CaracalStreamer for a CARACAL data root.

        Args:
            rootpath (str): A CARACAL session directory or a parent of sessions.
        """
        super().__init__()

        self.rootpath = rootpath
        self.syslog_files: list[str] = []

        self.register_syslog_files()

    def configuration_parameters(self) -> dict[str, any]:
        """Return the streamer's configuration parameters.

        Returns:
            dict: Contains 'rootpath'.
        """
        return {
            "rootpath": self.rootpath,
        }

    @staticmethod
    def discover_syslogs(rootpath: str) -> tuple[list[str], list[str]]:
        """Find sessions at the selected level and flag apparent missing syslogs.

        Returns (syslog paths, names of recording folders without syslogs).
        Unrelated folders are ignored; discovery is deliberately not recursive.
        """
        root = Path(rootpath).expanduser()
        if not root.is_dir():
            raise ValueError(f"CARACAL directory does not exist: {root}")

        if (root / "syslog.txt").is_file():
            return [str(root / "syslog.txt")], []

        folders = sorted((path for path in root.iterdir() if path.is_dir()))
        syslogs = []
        missing = []
        for folder in folders:
            syslog = folder / "syslog.txt"
            if syslog.is_file():
                syslogs.append(str(syslog))
            elif any(
                child.is_dir() and child.name.startswith("aud")
                and child.name[3:].isdigit()
                for child in folder.iterdir()
            ):
                missing.append(folder.name)

        # The root itself may be a recording folder with a missing syslog.
        if not syslogs and not missing and any(
            folder.name.startswith("aud") and folder.name[3:].isdigit()
            for folder in folders
        ):
            missing.append(root.name)

        return syslogs, missing

    def register_syslog_files(self) -> None:
        """Register available sessions; warn when recording folders lack logs."""
        self.syslog_files, self.missing_syslogs = self.discover_syslogs(
            self.rootpath
        )
        if self.missing_syslogs:
            print(
                "WARNING: CARACAL folders without syslog.txt will be skipped: "
                + ", ".join(self.missing_syslogs)
            )
        if not self.syslog_files:
            raise ValueError(
                f"No syslog.txt found in {self.rootpath} or its immediate "
                "session subdirectories."
            )

    def get_audio_packets(self, progress_callback=None) -> list[AudioPacket]:
        """
        Parse the registered CARACAL sessions and return them as
        generic AudioPacket objects.

        ``progress_callback`` is optional and receives ``(completed, total)``
        after each syslog.  GUI callers can use it as a lightweight heartbeat
        without coupling this source object to Qt.
        """

        packets: list[AudioPacket] = []
        total_syslogs = len(self.syslog_files)

        for syslog_index, syslog_file in enumerate(self.syslog_files, start=1):
            parser = SyslogParser(
                syslog_file
            )

            container = parser.process()

            for session_idx, session in enumerate(
                container.sessions
            ):
                header = session.header

                if not self.syslog_session_header_ok(
                    header
                ):
                    sys_duration = getattr(
                        header,
                        "sysDuration",
                        None,
                    )

                    stats = getattr(
                        header,
                        "stats",
                        None,
                    )

                    num_files = getattr(
                        stats,
                        "num_files",
                        None,
                    )

                    print(
                        f"WARNING: Skipping invalid CARACAL session "
                        f"{session_idx} in {syslog_file}. "
                        f"sysDuration={sys_duration}, "
                        f"num_files={num_files}"
                    )

                    continue

                packet = (
                    self.build_audio_packet_from_session(
                        session,
                        syslog_file,
                    )
                )

                packets.append(
                    packet
                )

            if progress_callback is not None:
                progress_callback((syslog_index, total_syslogs))

        return packets


    def stream(self) -> None:
        """
        Emit each available AudioPacket into the pipeline.
        """

        for packet in self.get_audio_packets():
            self.next_audio(
                packet
            )
                
    def build_audio_packet_from_session(self, session, syslog_file: str) -> AudioPacket:

        audio_paths = [
            os.path.join(os.path.dirname(syslog_file), audio_file.subpath)
            for audio_file in session.audioFiles
        ]

        return AudioPacket(
            audio_paths=audio_paths,
            offset=0.0,
            duration=session.header.sysDuration,
            is_whole=True,
            lat=session.header.stats.median_GPS_lat,
            lon=session.header.stats.median_GPS_lon,
            recording_id=self._recording_id(audio_paths, syslog_file),
            display_name=(
                f"{session.header.headerID.deviceID} "
            ),
            misc_metadata={
                "source": "CARACAL",
                "syslog_file": str(syslog_file),
                "session_header": session.header,
                "num_audio_files": len(session.audioFiles),
                "device_id": session.header.headerID.deviceID,
                "card_id" : session.header.headerID.cardID,                
            },
        )

    def syslog_session_header_ok(self, header) -> bool:
        """
        Check whether a CARACAL session header is valid.

        Returns False if sysDuration <= 0.0 or num_files <= 0.
        """
        return header.sysDuration > 0.0 and header.stats.num_files > 0 



    def _recording_id(self, audio_paths, syslog_file):
        # Include the session folder in the identity regardless of whether the
        # user selected that folder or its parent. Existing cache keys for
        # multi-session roots remain unchanged.
        identity_root = os.path.dirname(os.path.dirname(syslog_file))
        relative_paths = [
            os.path.relpath(path, identity_root)
            for path in audio_paths
        ]

        identity = "\n".join(
            relative_paths
        )

        return sha256(
            identity.encode("utf-8")
        ).hexdigest()
            
        
        
    
    
