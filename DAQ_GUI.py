"""
DAQ_GUI.py

GUI front-end for DAQbyBScache.py (BS-cache based data acquisition at SwissFEL).

Features
  - Two measurement modes:
        * Single acquisition  (the 'dummy' mode of DAQbyBScache.py)
        * Parameter scan      (step one or more PVs through linear ranges)
  - Channel selection from the predefined groups in DAQbyBScache.py
    (spectrometer, other PVs, seed laser, BPMs) + free extra channels
  - Athos bunch selection (1 / 2) for the bunch-dependent channel names
  - Slow (CA) channels, read once with caget, saved as 'environment data'
  - Acquisition runs in a worker thread: the GUI stays responsive,
    shows progress, and the run can be aborted at any time
    (data of completed steps is still saved)
  - Live plots of any acquired channel (scalar or spectrum-like)
  - HDF5 output with the same layout as DAQbyBScache.py:
        <base>/YYYY/MM/DD/spec_meas_by_bschannel/scanNNNN/scanNNNN_<comment>_<timestamp>.h5
"""

import sys
import os
import time
import datetime
import numpy as np
import h5py

import DAQbyBScache as daq
from DAQbyBScache import DaqByBSCache, epics

from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QMessageBox, QStatusBar, QProgressBar,
    QPlainTextEdit, QFileDialog, QDialog, QDialogButtonBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

################################################################
# Channel groups (taken from DAQbyBScache.py)
################################################################
SPECTROMETERS = {
    "None":        [],
    "PMOS Maloja": daq.PMOS_Maloja,
    "PSRD Furka":  daq.PSRD_Furka,
    "PMOS Furka":  daq.PMOS_Furka,
    "PSSS":        daq.PSSS,
}
DEFAULT_SLOW_CHS = [
    'SATFE10-PEPG046:ENERGY-IN-USE',
    'SATFE10-PEPG046:PHOTONS-PER-SECOND-US.B',
    'SATFE10-PEPG046:PHOTONS-PER-SECOND-US.J',
    'SATFE10-PEPG046:PHOTONS-PER-SECOND-DS.B',
    'SATFE10-PEPG046:PHOTONS-PER-SECOND-DS.J',
    'SATFE10-PEPG046:PHOTONS-PER-SECOND-DS.G',
    'SATFE10-PEPG046:PHOTONS-PER-SECOND-DS.D',
]
DEFAULT_BASE_DIR = '/sf/data/measurements'
# defaults = the current configuration of DAQbyBScache.py
DEFAULT_SPECTROMETER = "PSRD Furka"
DEFAULT_GROUPS = {"BPMs"}


def channel_groups(bunch2athos):
    """Return the predefined BS channel groups for the given Athos bunch."""
    return {
        "Other PVs":  list(daq.get_otherPV_chs(bunch2athos)),
        "Seed laser": list(daq.Laser_chs),
        "Laser cameras / timing": list(daq.otherPV_chs_continue),
        "BPMs":       list(daq.get_BPM_chs(bunch2athos)),
    }


def unique(seq):
    """Remove duplicates, keep order."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def make_output_filename(base_dir, comments=''):
    """Same naming convention as DAQbyBScache.setup_filename, but with a
    configurable base directory and creation of missing parent folders."""
    now = datetime.datetime.now()
    folder = os.path.join(base_dir, '%d/%02d/%02d' % (now.year, now.month, now.day),
                          'spec_meas_by_bschannel')
    os.makedirs(folder, exist_ok=True)
    n = len(os.listdir(folder)) + 1
    while os.path.exists(os.path.join(folder, 'scan%04d' % n)):
        n += 1
    folder = os.path.join(folder, 'scan%04d' % n)
    os.mkdir(folder)
    stamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    name = 'scan%04d_' % n + (comments + '_' if comments else '') + stamp + '.h5'
    return os.path.join(folder, name)


def to_array(values):
    """Convert a list of per-shot values to a numeric array (None -> NaN)."""
    try:
        return np.array(values, dtype=float)
    except (TypeError, ValueError):
        pass
    # some shots may be missing (None): try to fill with NaN of the right shape
    ref = next((np.asarray(v, dtype=float) for v in values if v is not None), None)
    if ref is None:
        return np.full(len(values), np.nan)
    filled = [np.asarray(v, dtype=float) if v is not None else np.full(ref.shape, np.nan)
              for v in values]
    return np.array(filled, dtype=float)


################################################################
# stdout -> GUI log
################################################################
class LogStream(QObject):
    text = pyqtSignal(str)

    def write(self, s):
        if sys.__stdout__:
            sys.__stdout__.write(s)
        self.text.emit(s)

    def flush(self):
        if sys.__stdout__:
            sys.__stdout__.flush()


################################################################
# Worker thread
################################################################
class DaqWorker(QThread):
    progress  = pyqtSignal(int, int)          # shots done, total shots
    status    = pyqtSignal(str)
    step_done = pyqtSignal(int, object, object)  # step index (in order of scan values), {ch: array}, {slow ch: value}
    finished_ok = pyqtSignal(str)             # output file
    failed    = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._abort = False

    def abort(self):
        self._abort = True

    @staticmethod
    def _read_slow(chs):
        slow = {}
        for ch in chs:
            try:
                v = epics.caget(ch)
                if v is None:
                    print('caget returned None for %s' % ch)
                    continue
                slow[ch] = np.array(v)
            except Exception as e:
                print('caget failed for %s: %s' % (ch, e))
        return slow

    # sleep that can be interrupted by abort
    def _wait(self, seconds):
        t_end = time.time() + seconds
        while time.time() < t_end and not self._abort:
            time.sleep(0.05)

    def run(self):
        cfg = self.cfg
        cache = None
        init_vals = []
        try:
            channels = cfg['channels']
            nshots   = cfg['nshots']
            is_scan  = cfg['mode'] == 'scan'
            scan_pvs = cfg['scan_pvs'] if is_scan else []
            ranges   = np.atleast_2d(cfg['scan_ranges']) if is_scan else np.zeros((0, 1))
            nsteps   = ranges.shape[1] if is_scan else 1
            total    = nsteps * nshots

            self.status.emit('Connecting to %d BS channels...' % len(channels))
            cache = DaqByBSCache(channels)
            cache.connect_all()

            # initial values of the scan PVs
            for pv in scan_pvs:
                init_vals.append(epics.caget(pv))
            if is_scan:
                print('Initial values of scan PVs:', init_vals)

            # scan order: start from the end closest to the current value (as in DAQbyBScache.py)
            order = list(range(nsteps))
            if is_scan and cfg['nearest_first'] and nsteps > 1 and init_vals[0] is not None:
                if abs(init_vals[0] - ranges[0][0]) > abs(init_vals[0] - ranges[0][-1]):
                    order = order[::-1]

            # as in DAQbyBScache.py: in a scan the slow channels (+ scan PV readbacks) are read at every step
            slow_chs = unique(list(cfg['slow_chs']) + list(scan_pvs))
            steps = {}       # i_scan -> {ch: array}
            slow_steps = {}  # i_scan -> {slow ch: value}
            done_shots = 0
            for n, i_scan in enumerate(order):
                if self._abort:
                    break
                if is_scan:
                    vals = ranges[:, i_scan]
                    print('Step %d/%d, set values: %s' % (n + 1, nsteps, str(vals)))
                    for pv, v in zip(scan_pvs, vals):
                        epics.caput(pv, float(v))
                    self.status.emit('Step %d/%d: waiting %.1f s for PVs to settle...'
                                     % (n + 1, nsteps, cfg['wait_time']))
                    self._wait(cfg['wait_time'])
                    if self._abort:
                        break
                    cache.flush()
                self.status.emit('Step %d/%d: acquiring %d shots...' % (n + 1, nsteps, nshots)
                                 if is_scan else 'Acquiring %d shots...' % nshots)
                raw = []
                for _ in range(nshots):
                    if self._abort:
                        break
                    raw.append(cache.read())
                    done_shots += 1
                    self.progress.emit(done_shots, total)
                if self._abort or not raw:
                    break
                sorted_data = {}
                for ch in raw[0].keys():
                    sorted_data[ch] = to_array([r.get(ch) for r in raw])
                if 'pid' in sorted_data and len(sorted_data['pid']) > 1:
                    dpid = np.nanmean(np.diff(sorted_data['pid']))
                    if dpid > 0:
                        print('DAQ rate: mean pulse-id step %.2f  ->  %.1f Hz' % (dpid, 100.0 / dpid))
                steps[i_scan] = sorted_data
                slow_steps[i_scan] = self._read_slow(slow_chs) if is_scan else {}
                self.step_done.emit(i_scan, sorted_data, slow_steps[i_scan])

            # slow channels (single acquisition: read once)
            if not is_scan:
                self.status.emit('Reading slow channels...')
                slow = self._read_slow(slow_chs)
            else:
                slow = slow_steps

            if not steps:
                self.failed.emit('Aborted before any data were taken - nothing saved.')
                return

            # save
            self.status.emit('Saving data...')
            fname = make_output_filename(cfg['base_dir'], cfg['comment'])
            self._save(fname, steps, slow, scan_pvs, ranges, init_vals, nsteps, nshots)
            print('Data saved in:', fname)
            self.finished_ok.emit(fname)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit('%s: %s' % (type(e).__name__, e))
        finally:
            if cfg['mode'] == 'scan' and cfg['return2init'] and init_vals:
                print('Returning scan PVs to initial values...')
                for pv, v in zip(cfg['scan_pvs'], init_vals):
                    if v is not None:
                        epics.caput(pv, v)
            if cache is not None:
                try:
                    cache.terminate()
                except Exception as e:
                    print('Error while stopping BS cache:', e)

    def _save(self, fname, steps, slow, scan_pvs, ranges, init_vals, nsteps, nshots):
        cfg = self.cfg
        with h5py.File(fname, 'w') as f:
            f.attrs['aborted'] = bool(self._abort)
            f.attrs['comment'] = cfg['comment']
            f.attrs['athos_bunch'] = cfg['bunch']
            f.attrs['created'] = datetime.datetime.now().isoformat()
            f.create_dataset('nshots', data=nshots)
            if cfg['mode'] != 'scan':
                f.create_dataset('scan_type', data=['dummy'])
                for ch, arr in steps[0].items():
                    try:
                        f.create_dataset(ch, data=arr)
                    except Exception as e:
                        print('could not save %s: %s' % (ch, e))
                env = f.create_group('environment data')
                for ch, v in slow.items():
                    try:
                        env.create_dataset(ch, data=v)
                    except Exception as e:
                        print('could not save %s: %s' % (ch, e))
            else:
                done = sorted(steps.keys())   # completed steps, in order of scan values
                f.create_dataset('scan_type', data=['scan'])
                f.create_dataset('nsteps', data=len(done))
                f.create_dataset('wait_time', data=cfg['wait_time'])
                for k, pv in enumerate(scan_pvs):
                    f.create_dataset(pv + '_scan', data=ranges[k, done])
                    if init_vals[k] is not None:
                        f.create_dataset(pv + '_init', data=init_vals[k])
                for ch in steps[done[0]].keys():
                    try:
                        f.create_dataset(ch, data=np.array([steps[i][ch] for i in done]))
                    except Exception as e:
                        print('could not save %s: %s' % (ch, e))
                # slow channels: one value per step, shape [nsteps]; saved in 'environment data'
                # and (as DAQbyBScache.py does) also at the top level
                env = f.create_group('environment data')
                slow_names = unique([ch for i in done for ch in slow[i].keys()])
                for ch in slow_names:
                    try:
                        vals = to_array([slow[i].get(ch) for i in done])
                        env.create_dataset(ch, data=vals)
                        if ch not in f:
                            f.create_dataset(ch, data=vals)
                    except Exception as e:
                        print('could not save %s: %s' % (ch, e))


################################################################
# Main window
################################################################
class DaqWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.worker = None
        self.results = None    # {'mode', 'scan_pvs', 'ranges', 'steps': {i: {ch: arr}}}
        self._build_ui()

        self._log_stream = LogStream()
        self._log_stream.text.connect(self._append_log)
        sys.stdout = self._log_stream

        self._update_mode()
        self._update_channel_count()

    # ── UI layout ────────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle("SwissFEL  ·  DAQ by BS cache")
        self.setGeometry(80, 80, 1450, 900)
        self.setStyleSheet(STYLESHEET)

        root = QWidget()
        root_v = QVBoxLayout(root)
        root_v.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._left_panel())
        scroll.setMinimumWidth(440)
        splitter.addWidget(scroll)
        splitter.addWidget(self._right_panel())
        splitter.setSizes([460, 990])
        root_v.addWidget(splitter)
        self.setCentralWidget(root)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready  –  configure the measurement and click  'Start'.")

    def _left_panel(self):
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(8)

        # ---- measurement ----
        grp = QGroupBox("Measurement")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_box = QComboBox()
        self.mode_box.addItems(["Single acquisition", "Parameter scan"])
        self.mode_box.currentIndexChanged.connect(self._update_mode)
        g.addWidget(self.mode_box, 0, 1)

        g.addWidget(QLabel("Shots per step:"), 1, 0)
        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(1, 100000)
        self.shots_spin.setValue(100)
        g.addWidget(self.shots_spin, 1, 1)

        g.addWidget(QLabel("Athos bunch:"), 2, 0)
        self.bunch_box = QComboBox()
        self.bunch_box.addItems(["1", "2"])
        self.bunch_box.setCurrentText(str(daq.Bunch2Athos))
        self.bunch_box.setToolTip("Selects the bunch-dependent channel names (…-B1 / …-B2, BPM X1 / X2).")
        self.bunch_box.currentIndexChanged.connect(self._update_channel_count)
        g.addWidget(self.bunch_box, 2, 1)

        g.addWidget(QLabel("Comment:"), 3, 0)
        self.comment_edit = QLineEdit("DAQtest")
        self.comment_edit.setToolTip("Added to the output file name.")
        g.addWidget(self.comment_edit, 3, 1)
        g.setColumnStretch(1, 1)
        vbox.addWidget(grp)

        # ---- BS channels ----
        grp = QGroupBox("BS channels")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Spectrometer:"), 0, 0)
        self.spec_box = QComboBox()
        self.spec_box.addItems(list(SPECTROMETERS.keys()))
        self.spec_box.setCurrentText(DEFAULT_SPECTROMETER)
        self.spec_box.currentIndexChanged.connect(self._update_channel_count)
        g.addWidget(self.spec_box, 0, 1)

        self.group_checks = {}
        for r, name in enumerate(channel_groups(1).keys(), start=1):
            cb = QCheckBox(name)
            cb.setChecked(name in DEFAULT_GROUPS)
            cb.stateChanged.connect(self._update_channel_count)
            self.group_checks[name] = cb
            g.addWidget(cb, r, 0, 1, 2)

        r = len(self.group_checks) + 1
        g.addWidget(QLabel("Extra channels (one per line):"), r, 0, 1, 2)
        self.extra_edit = QPlainTextEdit()
        self.extra_edit.setPlaceholderText("e.g.  SATFE10-PEPG046-EVR0:CALCI")
        self.extra_edit.setFixedHeight(64)
        self.extra_edit.textChanged.connect(self._update_channel_count)
        g.addWidget(self.extra_edit, r + 1, 0, 1, 2)

        self.lbl_nch = QLabel()
        self.lbl_nch.setObjectName("hint")
        g.addWidget(self.lbl_nch, r + 2, 0)
        show_btn = QPushButton("Show list…")
        show_btn.clicked.connect(self._show_channel_list)
        g.addWidget(show_btn, r + 2, 1, alignment=Qt.AlignRight)
        g.setColumnStretch(1, 1)
        vbox.addWidget(grp)

        # ---- slow channels ----
        grp = QGroupBox("Slow channels  (caget)")
        v = QVBoxLayout(grp)
        hint = QLabel("Single acquisition: read once.  Scan: read at every step, together with the scan PVs' readbacks.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.slow_edit = QPlainTextEdit("\n".join(DEFAULT_SLOW_CHS))
        self.slow_edit.setFixedHeight(90)
        v.addWidget(self.slow_edit)
        vbox.addWidget(grp)

        # ---- scan ----
        self.grp_scan = QGroupBox("Scan")
        v = QVBoxLayout(self.grp_scan)
        self.scan_table = QTableWidget(0, 3)
        self.scan_table.setHorizontalHeaderLabels(["PV name", "Start", "End"])
        hh = self.scan_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.scan_table.verticalHeader().setVisible(False)
        self.scan_table.setFixedHeight(110)
        self.scan_table.itemChanged.connect(self._update_scan_preview)
        v.addWidget(self.scan_table)

        row = QHBoxLayout()
        add_btn = QPushButton("+ Add PV")
        add_btn.clicked.connect(lambda: self._add_scan_row())
        del_btn = QPushButton("− Remove PV")
        del_btn.clicked.connect(self._del_scan_row)
        row.addWidget(add_btn)
        row.addWidget(del_btn)
        v.addLayout(row)

        g = QGridLayout()
        g.addWidget(QLabel("Number of steps:"), 0, 0)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 10000)
        self.steps_spin.setValue(5)
        self.steps_spin.valueChanged.connect(self._update_scan_preview)
        g.addWidget(self.steps_spin, 0, 1)
        g.addWidget(QLabel("Wait after setting (s):"), 1, 0)
        self.wait_spin = QDoubleSpinBox()
        self.wait_spin.setRange(0, 3600)
        self.wait_spin.setDecimals(1)
        self.wait_spin.setValue(10.0)
        self.wait_spin.valueChanged.connect(self._update_scan_preview)
        g.addWidget(self.wait_spin, 1, 1)
        g.setColumnStretch(1, 1)
        v.addLayout(g)

        self.nearest_check = QCheckBox("Start from the end closest to the current value")
        self.nearest_check.setChecked(False)
        v.addWidget(self.nearest_check)
        self.return_check = QCheckBox("Return to initial values when finished")
        self.return_check.setChecked(True)
        v.addWidget(self.return_check)
        self.lbl_scan_preview = QLabel()
        self.lbl_scan_preview.setObjectName("hint")
        self.lbl_scan_preview.setWordWrap(True)
        v.addWidget(self.lbl_scan_preview)
        self._add_scan_row()
        vbox.addWidget(self.grp_scan)

        # ---- output ----
        grp = QGroupBox("Output")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Base folder:"), 0, 0)
        self.dir_edit = QLineEdit(DEFAULT_BASE_DIR)
        g.addWidget(self.dir_edit, 0, 1)
        br = QPushButton("…")
        br.setFixedWidth(32)
        br.clicked.connect(self._browse_dir)
        g.addWidget(br, 0, 2)
        g.addWidget(QLabel("Last file:"), 1, 0)
        self.lbl_last = QLabel("—")
        self.lbl_last.setObjectName("hint")
        self.lbl_last.setWordWrap(True)
        self.lbl_last.setTextInteractionFlags(Qt.TextSelectableByMouse)
        g.addWidget(self.lbl_last, 1, 1, 1, 2)
        g.setColumnStretch(1, 1)
        vbox.addWidget(grp)

        # ---- actions ----
        row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.do_start)
        self.abort_btn = QPushButton("■  Abort")
        self.abort_btn.setObjectName("danger")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self.do_abort)
        row.addWidget(self.start_btn, 2)
        row.addWidget(self.abort_btn, 1)
        vbox.addLayout(row)

        vbox.addStretch()
        return panel

    def _right_panel(self):
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(6)

        # progress
        row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m shots")
        row.addWidget(self.progress)
        vbox.addLayout(row)

        # plot controls
        row = QHBoxLayout()
        row.addWidget(QLabel("Channel to plot:"))
        self.plot_ch_box = QComboBox()
        self.plot_ch_box.setMinimumWidth(380)
        self.plot_ch_box.currentIndexChanged.connect(self.redraw)
        row.addWidget(self.plot_ch_box, 1)
        open_btn = QPushButton("Open .h5…")
        open_btn.setToolTip("Load a previously saved DAQ file for plotting.")
        open_btn.clicked.connect(self._open_file)
        row.addWidget(open_btn)
        save_btn = QPushButton("Save figure…")
        save_btn.clicked.connect(self._save_figure)
        row.addWidget(save_btn)
        vbox.addLayout(row)

        # figure
        self._fig = Figure(figsize=(9, 4.5), layout="constrained", facecolor="white")
        self._canvas = FigureCanvas(self._fig)
        toolbar = NavToolbar(self._canvas, panel)
        self._ax = self._fig.subplots(1, 2)

        plot_split = QSplitter(Qt.Vertical)
        fig_w = QWidget()
        fv = QVBoxLayout(fig_w)
        fv.setContentsMargins(0, 0, 0, 0)
        fv.addWidget(toolbar)
        fv.addWidget(self._canvas)
        plot_split.addWidget(fig_w)

        log_w = QGroupBox("Log")
        lv = QVBoxLayout(log_w)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("log")
        self.log_view.setMaximumBlockCount(5000)
        lv.addWidget(self.log_view)
        plot_split.addWidget(log_w)
        plot_split.setSizes([620, 220])
        vbox.addWidget(plot_split, 1)

        self._draw_empty()
        return panel

    # ── helpers ──────────────────────────────────────────────────────
    def _append_log(self, s):
        self.log_view.moveCursor(QTextCursor.End)
        self.log_view.insertPlainText(s)
        self.log_view.moveCursor(QTextCursor.End)

    def _is_scan(self):
        return self.mode_box.currentIndex() == 1

    def _update_mode(self):
        self.grp_scan.setEnabled(self._is_scan())
        self._update_scan_preview()

    def _collect_channels(self):
        bunch = int(self.bunch_box.currentText())
        groups = channel_groups(bunch)
        chs = list(SPECTROMETERS[self.spec_box.currentText()])
        for name, cb in self.group_checks.items():
            if cb.isChecked():
                chs += groups[name]
        chs += [l.strip() for l in self.extra_edit.toPlainText().splitlines() if l.strip()]
        return unique(chs)

    def _update_channel_count(self):
        bunch = int(self.bunch_box.currentText())
        groups = channel_groups(bunch)
        for name, cb in self.group_checks.items():
            cb.setText("%s  (%d)" % (name, len(unique(groups[name]))))
        self.lbl_nch.setText("%d unique channels selected" % len(self._collect_channels()))

    def _show_channel_list(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Selected BS channels")
        dlg.resize(480, 600)
        v = QVBoxLayout(dlg)
        chs = self._collect_channels()
        v.addWidget(QLabel("%d channels" % len(chs)))
        t = QPlainTextEdit("\n".join(chs))
        t.setReadOnly(True)
        t.setObjectName("log")
        v.addWidget(t)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.exec_()

    def _add_scan_row(self, pv="", start="0", end="1"):
        self.scan_table.blockSignals(True)
        r = self.scan_table.rowCount()
        self.scan_table.insertRow(r)
        for c, txt in enumerate([pv, start, end]):
            self.scan_table.setItem(r, c, QTableWidgetItem(txt))
        self.scan_table.blockSignals(False)
        self._update_scan_preview()

    def _del_scan_row(self):
        r = self.scan_table.currentRow()
        if r < 0:
            r = self.scan_table.rowCount() - 1
        if r >= 0:
            self.scan_table.removeRow(r)
        self._update_scan_preview()

    def _collect_scan(self):
        """Return (pvs, ranges[n_pv, nsteps]); raises ValueError on bad input."""
        pvs, rng = [], []
        n = self.steps_spin.value()
        for r in range(self.scan_table.rowCount()):
            items = [self.scan_table.item(r, c) for c in range(3)]
            pv = items[0].text().strip() if items[0] else ""
            if not pv:
                continue
            try:
                lo = float(items[1].text())
                hi = float(items[2].text())
            except (AttributeError, ValueError):
                raise ValueError("Start/End of '%s' are not numbers." % pv)
            pvs.append(pv)
            rng.append(np.linspace(lo, hi, n))
        return pvs, np.array(rng).reshape(len(pvs), n)

    def _update_scan_preview(self, *args):
        if not hasattr(self, 'lbl_scan_preview'):
            return
        try:
            pvs, rng = self._collect_scan()
        except ValueError as e:
            self.lbl_scan_preview.setText("⚠ " + str(e))
            return
        if not pvs:
            self.lbl_scan_preview.setText("Add at least one PV to scan.")
            return
        n = rng.shape[1]
        vals = ", ".join("%.4g" % x for x in rng[0][:8]) + (", …" if n > 8 else "")
        t_est = n * (self.wait_spin.value() + self.shots_spin.value() / 100.0)
        self.lbl_scan_preview.setText(
            "%d PV(s), %d steps.  %s: [%s]\nEstimated duration ≈ %.0f s (assuming 100 Hz)."
            % (len(pvs), n, pvs[0], vals, t_est))

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Base output folder", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    # ── actions ──────────────────────────────────────────────────────
    def do_start(self):
        channels = self._collect_channels()
        if not channels:
            QMessageBox.warning(self, "No channels", "Select at least one BS channel.")
            return
        cfg = dict(
            mode='scan' if self._is_scan() else 'dummy',
            channels=channels,
            slow_chs=unique([l.strip() for l in self.slow_edit.toPlainText().splitlines() if l.strip()]),
            nshots=self.shots_spin.value(),
            bunch=int(self.bunch_box.currentText()),
            comment=self.comment_edit.text().strip().replace(' ', '_'),
            base_dir=self.dir_edit.text().strip() or DEFAULT_BASE_DIR,
            wait_time=self.wait_spin.value(),
            nearest_first=self.nearest_check.isChecked(),
            return2init=self.return_check.isChecked(),
            scan_pvs=[], scan_ranges=np.zeros((0, 1)),
        )
        if cfg['mode'] == 'scan':
            try:
                cfg['scan_pvs'], cfg['scan_ranges'] = self._collect_scan()
            except ValueError as e:
                QMessageBox.warning(self, "Scan setup", str(e))
                return
            if not cfg['scan_pvs']:
                QMessageBox.warning(self, "Scan setup", "Add at least one PV to scan.")
                return
            summary = ("Parameter scan: %d steps × %d shots\n\nScan PVs:\n" % (cfg['scan_ranges'].shape[1], cfg['nshots'])
                       + "\n".join("  %s:  %.4g → %.4g" % (pv, r[0], r[-1])
                                   for pv, r in zip(cfg['scan_pvs'], cfg['scan_ranges'])))
        else:
            summary = "Single acquisition: %d shots" % cfg['nshots']
        summary += "\n\n%d BS channels, %d slow channels\nOutput under: %s" % (
            len(channels), len(cfg['slow_chs']), cfg['base_dir'])
        if QMessageBox.question(self, "Start measurement?", summary,
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return

        print('\n=== %s  –  %s ===' % (datetime.datetime.now().strftime('%H:%M:%S'),
                                     'scan' if cfg['mode'] == 'scan' else 'single acquisition'))
        self.results = dict(mode=cfg['mode'], scan_pvs=cfg['scan_pvs'],
                            ranges=np.atleast_2d(cfg['scan_ranges']), steps={})
        nsteps = cfg['scan_ranges'].shape[1] if cfg['mode'] == 'scan' else 1
        self.progress.setMaximum(nsteps * cfg['nshots'])
        self.progress.setValue(0)

        self.worker = DaqWorker(cfg)
        self.worker.progress.connect(lambda d, t: self.progress.setValue(d))
        self.worker.status.connect(self.status.showMessage)
        self.worker.step_done.connect(self._on_step_done)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_thread_end)
        self._set_running(True)
        self.worker.start()

    def do_abort(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.abort()
            self.status.showMessage("Aborting – completed steps will be saved...")
            self.abort_btn.setEnabled(False)

    def _set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.abort_btn.setEnabled(running)
        for w in (self.mode_box, self.shots_spin, self.bunch_box, self.spec_box,
                  self.extra_edit, self.slow_edit, self.dir_edit, self.comment_edit):
            w.setEnabled(not running)
        for cb in self.group_checks.values():
            cb.setEnabled(not running)
        self.grp_scan.setEnabled(self._is_scan() and not running)

    def _on_step_done(self, i_scan, data, slow):
        if self.results['mode'] == 'scan' and slow:
            data = dict(data)
            for ch, v in slow.items():
                data.setdefault(ch, np.atleast_1d(np.asarray(v, dtype=float)))
        self.results['steps'][i_scan] = data
        self._refresh_channel_box(list(data.keys()))
        self.redraw()

    def _on_finished(self, fname):
        self.lbl_last.setText(fname)
        self.status.showMessage("Done  →  " + fname)

    def _on_failed(self, msg):
        self.status.showMessage("Measurement failed / aborted: " + msg)
        QMessageBox.warning(self, "Measurement", msg)

    def _on_thread_end(self):
        self._set_running(False)

    def _refresh_channel_box(self, chs):
        current = self.plot_ch_box.currentText()
        existing = [self.plot_ch_box.itemText(i) for i in range(self.plot_ch_box.count())]
        if existing == chs:
            return
        self.plot_ch_box.blockSignals(True)
        self.plot_ch_box.clear()
        self.plot_ch_box.addItems(chs)
        if current in chs:
            self.plot_ch_box.setCurrentText(current)
        else:
            # prefer a spectrum, then a calibrated intensity, then the first channel
            pref = [c for c in chs if c.endswith('SPECTRUM_Y')] + \
                   [c for c in chs if 'INTENSITY-CAL' in c]
            self.plot_ch_box.setCurrentText(pref[0] if pref else chs[0])
        self.plot_ch_box.blockSignals(False)

    def _open_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open DAQ file", self.dir_edit.text(), "HDF5 (*.h5 *.hdf5)")
        if not fname:
            return
        try:
            with h5py.File(fname, 'r') as f:
                stype = f['scan_type'][0]
                stype = stype.decode() if isinstance(stype, bytes) else str(stype)
                skip = {'scan_type', 'nshots', 'nsteps', 'wait_time'}
                chs = [k for k in f.keys() if isinstance(f[k], h5py.Dataset) and k not in skip
                       and not k.endswith('_scan') and not k.endswith('_init')]
                if stype == 'scan':
                    nst = int(f['nsteps'][()]) if 'nsteps' in f else len(f[chs[0]])
                    scan_pvs = [k[:-5] for k in f.keys() if k.endswith('_scan')
                                and np.ndim(f[k][()]) == 1 and len(f[k]) == nst]
                    if scan_pvs:
                        ranges = np.array([f[pv + '_scan'][()] for pv in scan_pvs])
                    else:   # old files: <PV>_scan contained the initial value -> plot vs step number
                        scan_pvs, ranges = ['step'], np.arange(nst, dtype=float)[None, :]
                    chs = [ch for ch in chs if len(f[ch].shape) > 0 and f[ch].shape[0] == nst]
                    steps = {i: {ch: f[ch][()][i] for ch in chs} for i in range(nst)}
                    self.results = dict(mode='scan', scan_pvs=scan_pvs, ranges=ranges, steps=steps)
                else:
                    self.results = dict(mode='dummy', scan_pvs=[], ranges=np.zeros((0, 1)),
                                        steps={0: {ch: f[ch][()] for ch in chs}})
            print('Loaded', fname)
            self._refresh_channel_box(chs)
            self.redraw()
            self.status.showMessage("Loaded  " + fname)
        except Exception as e:
            QMessageBox.warning(self, "Open file", "Could not read file:\n%s" % e)

    def _save_figure(self):
        start = self.lbl_last.text()
        start = os.path.splitext(start)[0] + '.png' if start != '—' else 'daq_plot.png'
        fname, _ = QFileDialog.getSaveFileName(self, "Save figure", start, "PNG (*.png);;PDF (*.pdf)")
        if fname:
            self._fig.savefig(fname, dpi=150)
            print('Figure saved:', fname)

    # ── plotting ─────────────────────────────────────────────────────
    def _style_axes(self):
        for ax in self._ax:
            ax.set_facecolor("white")
            ax.grid(True, color="#e5e7eb", lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#9ca3af")
            ax.tick_params(colors="#374151", labelsize=9)
            ax.title.set_fontsize(10)

    def _reset_axes(self):
        self._fig.clear()
        self._ax = self._fig.subplots(1, 2)
        self._style_axes()

    def _draw_empty(self):
        self._reset_axes()
        self._ax[0].set_title("Per-shot data")
        self._ax[1].set_title("Statistics")
        for ax in self._ax:
            ax.text(0.5, 0.5, "no data yet", ha='center', va='center',
                    color="#9ca3af", transform=ax.transAxes)
        self._canvas.draw_idle()

    def _xaxis_for(self, ch, step):
        """x-axis for spectrum-like channels: use the matching SPECTRUM_X if present."""
        if ch.endswith('SPECTRUM_Y'):
            xch = ch[:-1] + 'X'
            if xch in step:
                x = np.asarray(step[xch])
                if x.ndim == 2:
                    x = np.nanmean(x, axis=0)
                return x, 'Photon energy (eV)'
        return None, 'Pixel'

    def redraw(self):
        if not self.results or not self.results['steps']:
            self._draw_empty()
            return
        ch = self.plot_ch_box.currentText()
        if not ch:
            return
        self._reset_axes()
        try:
            if self.results['mode'] == 'scan':
                self._draw_scan(ch)
            else:
                self._draw_single(ch)
        except Exception as e:
            self._ax[0].text(0.5, 0.5, "cannot plot: %s" % e, ha='center', va='center',
                             color="#b91c1c", transform=self._ax[0].transAxes)
        self._canvas.draw_idle()

    def _draw_single(self, ch):
        step = self.results['steps'][0]
        arr = np.asarray(step[ch], dtype=float)
        a0, a1 = self._ax
        if arr.ndim == 1:
            shots = np.arange(1, len(arr) + 1)
            a0.plot(shots, arr, 'o-', color=BLUE, ms=3, lw=1)
            a0.set_xlabel("Shot")
            a0.set_ylabel(ch, fontsize=8)
            a0.set_title("Per-shot values")
            good = arr[np.isfinite(arr)]
            if len(good):
                a1.hist(good, bins=min(50, max(5, len(good) // 3)), color=BLUE, alpha=0.75, edgecolor='white')
                m, s = np.mean(good), np.std(good)
                a1.axvline(m, color=ORANGE, lw=1.5)
                a1.set_title("mean = %.4g,  std = %.4g  (%.1f %%)" % (m, s, 100 * s / abs(m) if m else np.nan))
            a1.set_xlabel(ch, fontsize=8)
            a1.set_ylabel("Counts")
        else:
            arr2 = arr.reshape(arr.shape[0], -1)
            x, xlabel = self._xaxis_for(ch, step)
            if x is None or len(x) != arr2.shape[1]:
                x, xlabel = np.arange(arr2.shape[1]), 'Pixel'
            ext = [x[0], x[-1], arr2.shape[0] + 0.5, 0.5]
            im = a0.imshow(arr2, aspect='auto', cmap='viridis', extent=ext, interpolation='nearest')
            self._fig.colorbar(im, ax=a0, pad=0.01)
            a0.grid(False)
            a0.set_xlabel(xlabel)
            a0.set_ylabel("Shot")
            a0.set_title("Waterfall")
            m = np.nanmean(arr2, axis=0)
            s = np.nanstd(arr2, axis=0)
            a1.fill_between(x, m - s, m + s, color=BLUE, alpha=0.18, lw=0, label='± std')
            a1.plot(x, m, color=BLUE, lw=1.4, label='mean')
            a1.set_xlabel(xlabel)
            a1.set_ylabel("Intensity (a.u.)")
            a1.set_title("Mean over %d shots" % arr2.shape[0])
            a1.legend(loc='upper right', fontsize=8, frameon=False)
        self._fig.suptitle(ch, fontsize=10, color="#111827")

    def _draw_scan(self, ch):
        res = self.results
        done = sorted(res['steps'].keys())
        xvals = res['ranges'][0, done]
        xname = res['scan_pvs'][0] if res['scan_pvs'] else 'step'
        a0, a1 = self._ax
        arrs = [np.atleast_1d(np.asarray(res['steps'][i][ch], dtype=float)) for i in done]
        if arrs[0].ndim == 1:
            for xv, a in zip(xvals, arrs):
                a0.plot(np.full(len(a), xv), a, '.', color=BLUE, alpha=0.25, ms=4)
            m = np.array([np.nanmean(a) for a in arrs])
            s = np.array([np.nanstd(a) for a in arrs])
            a0.errorbar(xvals, m, yerr=s, fmt='o-', color=ORANGE, ms=5, lw=1.4, capsize=3, label='mean ± std')
            a0.set_xlabel(xname, fontsize=8)
            a0.set_ylabel(ch, fontsize=8)
            a0.set_title("Value vs scan (%d/%d steps)" % (len(done), res['ranges'].shape[1]))
            a0.legend(loc='best', fontsize=8, frameon=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = 100 * s / np.abs(m)
            a1.plot(xvals, rel, 's-', color=GREEN, ms=4, lw=1.2)
            a1.set_xlabel(xname, fontsize=8)
            a1.set_ylabel("std / |mean|  (%)")
            a1.set_title("Relative shot-to-shot fluctuation")
        else:
            means = np.array([np.nanmean(a.reshape(a.shape[0], -1), axis=0) for a in arrs])
            x, xlabel = self._xaxis_for(ch, res['steps'][done[0]])
            if x is None or len(x) != means.shape[1]:
                x, xlabel = np.arange(means.shape[1]), 'Pixel'
            if len(done) > 1:
                ext = [x[0], x[-1], xvals[-1], xvals[0]]
                im = a0.imshow(means, aspect='auto', cmap='viridis', extent=ext, interpolation='nearest')
                self._fig.colorbar(im, ax=a0, pad=0.01)
                a0.grid(False)
            else:
                a0.plot(x, means[0], color=BLUE)
            a0.set_xlabel(xlabel)
            a0.set_ylabel(xname, fontsize=8)
            a0.set_title("Mean spectrum vs scan value")
            import matplotlib
            cmap = matplotlib.colormaps['viridis']
            sel = np.unique(np.linspace(0, len(done) - 1, min(len(done), 10)).astype(int))
            for k in sel:
                a1.plot(x, means[k], color=cmap(k / max(1, len(done) - 1)), lw=1.2,
                        label="%.4g" % xvals[k])
            a1.set_xlabel(xlabel)
            a1.set_ylabel("Intensity (a.u.)")
            a1.set_title("Mean spectra")
            a1.legend(loc='upper right', fontsize=7, frameon=False, title=xname, title_fontsize=7)
        self._fig.suptitle(ch, fontsize=10, color="#111827")

    # ── shutdown ─────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            if QMessageBox.question(self, "Measurement running",
                                    "A measurement is running. Abort it and quit?",
                                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.abort()
            self.worker.wait(30000)
        sys.stdout = sys.__stdout__
        event.accept()


################################################################
# White theme
################################################################
BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#16a34a"

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #ffffff;
    color: #1f2937;
    font-size: 10pt;
}
QScrollArea { border: none; }
QGroupBox {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    margin-top: 14px;
    padding: 8px 6px 6px 6px;
    background-color: #ffffff;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #1d4ed8;
}
QGroupBox:disabled::title { color: #9ca3af; }
QLabel { background: transparent; font-weight: normal; }
QLabel#hint { color: #6b7280; font-size: 9pt; }
QLabel:disabled { color: #9ca3af; }
QLineEdit, QPlainTextEdit, QTableWidget {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 3px 5px;
    selection-background-color: #bfdbfe;
    selection-color: #111827;
    font-weight: normal;
}
QLineEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #3b82f6;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTableWidget:disabled {
    background-color: #f9fafb;
    color: #9ca3af;
}
QSpinBox, QDoubleSpinBox, QComboBox { padding: 3px 5px; font-weight: normal; }
QPlainTextEdit#log {
    font-family: "DejaVu Sans Mono", "Consolas", monospace;
    font-size: 9pt;
    background-color: #f9fafb;
}
QHeaderView::section {
    background-color: #f3f4f6;
    color: #374151;
    border: none;
    border-bottom: 1px solid #e5e7eb;
    padding: 4px;
    font-weight: bold;
}
QTableWidget { gridline-color: #e5e7eb; }
QPushButton {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 5px;
    padding: 5px 12px;
    color: #1f2937;
    font-weight: normal;
}
QPushButton:hover   { background-color: #f3f4f6; }
QPushButton:pressed { background-color: #e5e7eb; }
QPushButton:disabled { color: #9ca3af; background-color: #f9fafb; border-color: #e5e7eb; }
QPushButton#primary {
    background-color: #2563eb; border-color: #2563eb; color: #ffffff;
    font-weight: bold; padding: 8px 12px;
}
QPushButton#primary:hover { background-color: #1d4ed8; }
QPushButton#primary:disabled { background-color: #93c5fd; border-color: #93c5fd; color: #ffffff; }
QPushButton#danger {
    background-color: #ffffff; border-color: #dc2626; color: #dc2626;
    font-weight: bold; padding: 8px 12px;
}
QPushButton#danger:hover { background-color: #fef2f2; }
QPushButton#danger:disabled { color: #fca5a5; border-color: #fecaca; }
QCheckBox { background: transparent; font-weight: normal; }
QCheckBox:disabled { color: #9ca3af; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #9ca3af; border-radius: 3px; background-color: #ffffff;
}
QCheckBox::indicator:hover { border-color: #2563eb; }
QCheckBox::indicator:checked { background-color: #2563eb; border-color: #2563eb; }
QCheckBox::indicator:disabled { background-color: #f3f4f6; border-color: #d1d5db; }
QCheckBox::indicator:checked:disabled { background-color: #93c5fd; border-color: #93c5fd; }
QProgressBar {
    border: 1px solid #d1d5db;
    border-radius: 5px;
    background-color: #f3f4f6;
    text-align: center;
    height: 18px;
}
QProgressBar::chunk { background-color: #3b82f6; border-radius: 4px; }
QStatusBar { background-color: #f9fafb; border-top: 1px solid #e5e7eb; color: #374151; }
QSplitter::handle { background-color: #f3f4f6; }
QToolTip { background-color: #ffffff; color: #1f2937; border: 1px solid #d1d5db; }
"""


def white_palette():
    from PyQt5.QtGui import QPalette, QColor
    pal = QPalette()
    for role, col in [(QPalette.Window, "#ffffff"), (QPalette.Base, "#ffffff"),
                      (QPalette.AlternateBase, "#f9fafb"), (QPalette.Button, "#ffffff"),
                      (QPalette.Text, "#1f2937"), (QPalette.WindowText, "#1f2937"),
                      (QPalette.ButtonText, "#1f2937"), (QPalette.Highlight, "#2563eb"),
                      (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#ffffff"),
                      (QPalette.ToolTipText, "#1f2937")]:
        pal.setColor(role, QColor(col))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#9ca3af"))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#9ca3af"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#9ca3af"))
    return pal


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(white_palette())
    window = DaqWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
