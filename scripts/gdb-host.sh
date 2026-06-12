#!/bin/sh
exec flatpak-spawn --host /usr/bin/gdb "$@"
