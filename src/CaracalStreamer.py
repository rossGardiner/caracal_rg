# CaracalStreamer
# This is an object which takes a Caracal Data directory and creates AudioPacket instances from each session in the directory. 
# This code is built to interface caracal-python with the rest of this programme
# CaracalStreamer inherits behaviour from the PipelineLink object
# To init a CaracalStreamer, you just need to pass a top level directory where sub-directories contain data from caracal sessions. The logic inside will visit each session's log file and create an AudioPacket instance for each, these will be called back to whatever callback you registered. 
# Note, due to inheritance, it is technically possible to pipe AudioPackets into CaracalStreamer, but this implementation doesnt make use of that feature. 

import caracal 
from glob import glob
import os 


from src.PipelineLink import PipelineLink


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

    def syslog_session_header_ok(self, header) -> bool:
        """
        Check whether a CARACAL session header is valid.

        Returns False if sysDuration <= 0.0 or num_files <= 0.
        """
        return header.sysDuration > 0.0 and header.stats.num_files > 0 
    
