#!/usr/bin/env python
"""
A script for updating the version number within a project. In source files, a
global attribute should be defined called VERSION. For README.md files,
include the following lines:
    *Current version:*
    *Updated on:*

To run, pass the version number as the sole argument to this script.
"""

import sys
import os
import datetime

#SRC_FILES = ['gfft.py', 'setup.py']  # Alter for each project...
SRC_FILES = ['setup.py']  # Alter for each project...

# These files should exist in ALL projects.
STANDARD_FILES = ['README.md']

vnum = sys.argv[1]

print vnum

for i in range(len(SRC_FILES)):
    # backup the file before writing over it.
    os.rename(SRC_FILES[i], SRC_FILES[i] + '.bak')
    outf = open(SRC_FILES[i], 'w')

    for line in open(SRC_FILES[i] + '.bak'):
        if "VERSION" in line:
            outf.write("VERSION = " + vnum + "\n")
        else:
            outf.write(line)

for i in range(len(STANDARD_FILES)):
    os.rename(STANDARD_FILES[i], STANDARD_FILES[i] + '.bak')
    outf = open(STANDARD_FILES[i], 'w')

    for line in open(STANDARD_FILES[i]):
        if "*current version:*" in line.lower():
            outf.write("*Current version: " + vnum + "\n")
        elif "*updated on:*" in line.lower():
            outf.write("*Updated on:* " + str(datetime.date.today()) + "/n")
        else:
            outf.write(line)
