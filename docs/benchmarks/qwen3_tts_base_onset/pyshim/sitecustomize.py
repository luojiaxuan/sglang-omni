# note (luojiaxuan): pins every process title on the shared host to a generic interpreter path.
import setproctitle as _setproctitle

_TITLE = "/usr/bin/python"
_real_setproctitle = _setproctitle.setproctitle
_real_setproctitle(_TITLE)
_setproctitle.setproctitle = lambda *_args, **_kwargs: _real_setproctitle(_TITLE)
