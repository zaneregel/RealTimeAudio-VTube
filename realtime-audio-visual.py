import numpy as np
import pyaudio
import wave
import sys
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox

# Constants for PyAudio
CHUNK = 2048  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Format of the audio (16-bit int)
CHANNELS = 2  # Mono audio
RATE = 48000  # Sampling rate

# Audio Frequency filtering levels
LOWER_FREQ = 10000
UPPER_FREQ = 10000

# Initialize PyAudio
p = pyaudio.PyAudio()

# Print available audio streams
#for i in range(p.get_device_count()):
#    info = p.get_device_info_by_index(i)
#    print(f"Device {i}: {info['name']}")
#    print(f"  Channels: {info['maxInputChannels']} input / {info['maxOutputChannels']} output")
#    print(f"  Default Sample Rate: {info['defaultSampleRate']}")
#    print(f"  Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")
#    print()

# Open audio stream
def initialize_mic():
    return p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=37,
                frames_per_buffer=CHUNK)

def initialize_browser():
    return p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=42,
                frames_per_buffer=CHUNK)

def initialize_desktop():
    return p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=40,
                frames_per_buffer=CHUNK)

current_stream = None
current_view = 'spectrum' #Default view

def switch_audio_source(source_id):
    global current_stream
    if current_stream:
        current_stream.stop_stream()
        current_stream.close()
    if source_id == 1:
        current_stream = initialize_mic()
    elif source_id == 2:
        current_stream = initialize_browser()
    elif source_id == 3:
        current_stream = initialize_desktop()
    current_stream.start_stream()

def update_view(view):
    global current_view
    current_view = view
    if view == "spectrum":
        spectrum_plot.show()
        signal_plot.hide()
    else:
        spectrum_plot.hide()
        signal_plot.show()

# Start app
app = QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True)

# Create Frequency Spectrum Plot
spectrum_plot = win.addPlot(row=0, col=0)
spectrum_curve = spectrum_plot.plot(pen=pg.mkPen(color="y", width=2))
#spectrum_plot.setLabel('left', 'Magnitude')
#spectrum_plot.setLabel('bottom', 'Frequency')
spectrum_plot.hideAxis('left')
spectrum_plot.hideAxis('bottom')
spectrum_plot.setRange(yRange=[0, 80000])  # Adjust as needed


# Create Raw Signal Plot
signal_plot = win.addPlot(row=1, col=0)
signal_curve = signal_plot.plot(pen=pg.mkPen(color="y", width=2))
#signalplot.setLabel('left', 'Amplitude')
#signalplot.setLabel('bottom', 'Sample Number')
signal_plot.hideAxis('left')
signal_plot.hideAxis('bottom')
signal_plot.setRange(xRange=[0, CHUNK], yRange=[-32768, 32767])
signal_plot.hide()   # hide initially as spectrum shows first

# Frequency scaling function
def scale_frequencies(freqs, lower_bound, upper_bound):
    """Scale frequencies below lower_bound and above upper_bound smoothly."""
    scale_factor = np.ones_like(freqs)
    
    # Linear scaling for lower frequencies
    low_mask = freqs < lower_bound
    scale_factor[low_mask] = np.interp(freqs[low_mask], [0, lower_bound], [0.0001, 1])
    
    # Linear scaling for upper frequencies
    high_mask = freqs > upper_bound
    scale_factor[high_mask] = np.interp(freqs[high_mask], [upper_bound, RATE / 2], [1, 0.0001])
    
    return scale_factor

# Apply Moving Average for Smoothing
def smooth(data, window_size):
    """Apply a moving average for smoothing."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# Display only one plot at a time
def update():
    if current_stream:
        data = np.frombuffer(current_stream.read(CHUNK), dtype=np.int16)
        fft_data = np.fft.fft(data)
        fft_freqs = np.fft.fftfreq(len(data), 1.0 / RATE)

        # Scale down low and high frequencies
        scaling = scale_frequencies(np.abs(fft_freqs), LOWER_FREQ, UPPER_FREQ)
        scaled_fft_data = fft_data * scaling

        fft_magnitudes = np.abs(scaled_fft_data)
        
        # Only keep the positive frequencies
        #pos_freqs = fft_freqs[:CHUNK // 2]
        #pos_magnitudes = fft_magnitudes[:CHUNK // 2]
        
        # Update Frequency Spectrum Plot
        if current_view == 'spectrum':
            smoothed_magnitude = smooth(fft_magnitudes, window_size=25)
            spectrum_curve.setData(fft_freqs, smoothed_magnitude)
            exporter = pg.exporters.ImageExporter(spectrum_plot.graphicsItem().scene())
            exporter.export("plot_img.png")
        
        # Update Signal Plot
        elif current_view == 'signal':
            data = np.frombuffer(current_stream.read(CHUNK), dtype=np.int16)
            signal_curve.setData(data)
            exporter = pg.exporters.ImageExporter(signal_plot.graphicsItem().scene())
            exporter.export("plot_img.png")

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        button1 = QPushButton("Switch to Mic (AT2020+)")
        button1.clicked.connect(lambda: switch_audio_source(1))
        layout.addWidget(button1)

        #button2 = QPushButton("Switch to Browser (VAIO3)")
        #button2.clicked.connect(lambda: switch_audio_source(2))
        #layout.addWidget(button2)

        button3 = QPushButton("Switch to Desktop (VAIO)")
        button3.clicked.connect(lambda: switch_audio_source(3))
        layout.addWidget(button3)

        combo = QComboBox()
        combo.addItem("Frequency Spectrum")
        combo.addItem("Raw Signal")
        combo.currentIndexChanged.connect(lambda index: update_view('spectrum' if index == 0 else 'signal'))
        layout.addWidget(combo)

        self.setLayout(layout)

main_window = MainWindow()
main_window.show()

sys.exit(app.exec_())