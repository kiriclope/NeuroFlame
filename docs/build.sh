#!/bin/sh
set -x
emacs --batch --no-init-file --directory $PWD --load ./docs/publish.el --funcall toggle-debug-on-error --funcall lc-publish-all
