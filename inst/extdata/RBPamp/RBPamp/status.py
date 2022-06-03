# -*- coding: future_fstrings -*-
from __future__ import absolute_import
from __future__ import print_function
__license__ = "MIT"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import os
import datetime
import logging

class StateTracker(object):
    def __init__(self, run, stage, startup=True):
        self.run = run
        self.rbp = run.rbp_name
        self.stage = stage
        self.fpath = os.path.join(self.run.run_path, '{}.txt'.format((stage)))
        self.state_file = open(self.fpath, 'a')
        self.logger = logging.getLogger('main.StateTracker({})'.format((stage)))
        self.completed_ts = None
        if startup:
            self.set("starting up")

    def set(self, state):
        ts = datetime.datetime.now().isoformat(' ')
        cols = [self.rbp, self.stage, ts, state, self.run.version, self.run.git_commit, self.run.cmdline]
        out = "\t".join([str(o) for o in cols])

        self.state_file.write(out + '\n')
        self.state_file.flush()
        self.logger.info('set "{}"'.format((out)))
    
    def is_completed(self, strict=False):
        lines = open(self.fpath, 'r').readlines()
        if not lines:
            return False

        last_state = lines[-1]

        try:
            rbp, stage, ts, state, version, git, cmdline = last_state.rstrip().split('\t')
        except ValueError:
            return False

        same_rbp = rbp == self.run.rbp_name
        same_stage = stage == self.stage
        same_version = version == self.run.version
        same_git = git == self.run.git_commit

        if not same_rbp and same_stage:
            raise ValueError("RBP or stage mismatch")
        
        if not (same_version and same_git):
            self.logger.warning("results computed with: {}-{} but currently at {}-{}".format((version), (git), (self.run.version), (self.run.git_commit)))
        
            if strict:
                return False
        
        if state.upper().startswith("COMPLETED"):
            self.completed_ts = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
            return True
        else:
            return False


