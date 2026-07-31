# CaracalStreamer
# This is an object which takes a Caracal Data directory and creates AudioPacket instances from each session in the directory. 
# This code is built to interface caracal-python with the rest of this programme
# CaracalStreamer inherits behaviour from the PipelineLink object
# To init a CaracalStreamer, you just need to pass a top level directory where sub-directories contain data from caracal sessions. The logic inside will visit each session's log file and create an AudioPacket instance for each, these will be called back to whatever callback you registered. 
# Note, due to inheritance, it is technically possible to pipe AudioPackets into CaracalStreamer, but this implementation doesnt make use of that feature. 

import os
from glob import glob

from caracal import SyslogParser

from src.PipelineLink import PipelineLink
from src.AudioPacket import AudioPacket


class CaracalStreamer(PipelineLink):
    """
    Takes a CARACAL data directory and creates AudioPacket instances from each
    session in the directory.

    The top-level directory is expected to contain one subdirectory per CARACAL
    session. Each session directory should contain a syslog.txt file.
    """

    def __init__(self, rootpath: str):
        super().__init__()

        self.rootpath = rootpath
        self.syslog_files: list[str] = []

        self.register_syslog_files()

    def register_syslog_files(self) -> None:
        """
        Search rootpath for syslog files.

        One syslog.txt is expected for each immediate subdirectory.
        """
        nr_expected_logs = sum(
            os.path.isdir(os.path.join(self.rootpath, p))
            for p in glob("*", root_dir=self.rootpath)
        )

        matches = [
            os.path.join(self.rootpath, p)
            for p in glob("*/syslog.txt", root_dir=self.rootpath)
            if os.path.isfile(os.path.join(self.rootpath, p))
        ]

        if len(matches) != nr_expected_logs:
            print(
                f"WARNING: Expected {nr_expected_logs} syslogs, found {len(matches)}. "
                "CaracalStreamer will continue with only the syslogs found. "
                "Audio from directories with missing syslogs will not be carried forward."
            )

        if matches:
            self.syslog_files = matches
        else:
            raise Exception(
                "No syslogs found! Are you sure this is a valid CARACAL data directory?"
            )
            
    def stream(self) -> None:
        for syslog_file in self.syslog_files:
            parser = SyslogParser(syslog_file)
            container = parser.process()

            for session_idx, session in enumerate(container.sessions):
                header = session.header

                if not self.syslog_session_header_ok(header):
                    sys_duration = getattr(header, "sysDuration", None)
                    stats = getattr(header, "stats", None)
                    num_files = getattr(stats, "num_files", None)

                    print(
                        f"WARNING: Skipping invalid CARACAL session "
                        f"{session_idx} in {syslog_file}. "
                        f"sysDuration={sys_duration}, "
                        f"num_files={num_files}"
                    )
                    continue

                packet = self.build_audio_packet_from_session(session, syslog_file)
                self.next_audio(packet)
                
    def build_audio_packet_from_session(self, session, syslog_file: str) -> AudioPacket:

        audio_paths = [
            os.path.join(self.rootpath, session.path, audio_file.subpath)
            for audio_file in session.audioFiles
        ]

        return AudioPacket(
            audio_paths=audio_paths,
            offset=0.0,
            duration=session.header.sysDuration,
            is_whole=True,
            lat=session.header.stats.median_GPS_lat,
            lon=session.header.stats.median_GPS_lon,
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
        
        
        
    
    
