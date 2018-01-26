# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of gwemopt
#
# gwemopt is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gwemopt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gwemopt.  If not, see <http://www.gnu.org/licenses/>

"""Git version generator
"""

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Adam Mercer <adam.mercer@ligo.org>'

import os
import subprocess
import time

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring

class GitStatus(object):
    """Git repository version information
    """
    def __init__(self):
        self._bin = self._find_git()
        self.id = None
        self.date = None
        self.branch = None
        self.tag = None
        self.author = None
        self.committer = None
        self.status = None

    # ------------------------------------------------------------------------
    # Core methods

    @staticmethod
    def _find_git():
        """Determine the full path of the git binary on this
        host
        """
        for path in os.environ['PATH'].split(os.pathsep):
            gitbin = os.path.join(path, 'git')
            if os.path.isfile(gitbin) and os.access(gitbin, os.X_OK):
                return gitbin
        raise ValueError("Git binary not found on this host")

    def git(self, *args):
        """Executable a command with arguments in a sub-process
        """
        cmdargs = [self._bin] + list(args)
        p = subprocess.Popen(cmdargs,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=isinstance(args, basestring))
        out, err = p.communicate()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode,
                                                ' '.join(cmdargs))
        return out.strip()

    # ------------------------------------------------------------------------
    # Git communication methods

    def get_commit_info(self):
        """Determine basic info about the latest commit
        """
        a, b, c, d, e, f =  self.git(
            'log', '-1', '--pretty=format:%H,%ct,%an,%ae,%cn,%ce').decode().split(',')
        self.id = a
        self.udate = b
        author = c
        author_email = d
        committer = e
        committer_email = f
        self.date = time.strftime('%Y-%m-%d %H:%M:%S +0000',
                                  time.gmtime(float(self.udate)))
        self.author = '%s <%s>' % (author, author_email)
        self.committer = '%s <%s>' % (committer, committer_email)

    def get_branch(self):
        branch = self.git('rev-parse', '--symbolic-full-name', 'HEAD').decode('utf-8')
        if branch == 'HEAD':
            self.branch = None
        else:
            self.branch = os.path.basename(branch)
        return self.branch

    def get_status(self):
        """Determine modification status of working tree
        """
        try:
            status = self.git('diff-files', '--quiet')
        except subprocess.CalledProcessError:
            self._status = 'UNCLEAN: Modified working tree'
        else:
            try:
                status = self.git('diff-index', '--cache', '--quiet',
                                  'HEAD')
            except subprocess.CalledProcessError:
                self.status = 'UNCLEAN: Modified working tree'
            else:
                self.status = 'CLEAN: All modifications committed'
        return self.status

    def get_tag(self):
        """Determine name of the current tag
        """
        if not self.id:
            self.get_commit_info()
        try:
            self.tag = self.git('describe', '--exact-match', '--tags',
                                 self.id)
        except subprocess.CalledProcessError:
            self.tag = None
        return self.tag

    # ------------------------------------------------------------------------
    # Write

    def write(self, fobj):
        """Write the contents of this `GitStatus` to a version.py format
        file object
        """
        # write file header
        fobj.write("# -*- coding: utf-8 -*-\n"
                   "# Copyright (C) Duncan Macleod (2013)\n\n"
                   "\"\"\"Versioning record for CIS\n\"\"\"\n\n")

        # write standard pythonic metadata
        fobj.write("__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'\n"
                   "__version__ = '%s'\n"
                   "__date__ = '%s'\n\n" % (self.version, self.date))

        # write git information
        for attr in ['id', 'branch', 'tag', 'author', 'committer', 'status']:
            val = getattr(self, attr)
            if val:
                fobj.write("git_%s = '%s'\n" % (attr, val))
            else:
                fobj.write("git_%s = None\n" % attr)

    def __call__(self, outputfile='version.py'):
        """Process the version information into a new file

        Parameters
        ----------
        outputfile : `str`
            path to output python file in which to write version info

        Returns
        -------
        info : `str`
            returns a string dump of the contents of the outputfile
        """
        self.get_commit_info()
        self.get_branch()
        self.get_tag()
        self.get_status()
        self.version = self.tag or self.id
        with open(outputfile, 'w') as fobj:
            self.write(fobj)
        with open(outputfile, 'r') as fobj:
            return fobj.read()
