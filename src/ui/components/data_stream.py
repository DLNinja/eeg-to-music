import socket
from PyQt5.QtCore import pyqtSignal, QThread

# Default values (used as UI defaults, can be changed by user)
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8888
DEFAULT_CHANNELS = 64
DEFAULT_SAMPLE_RATE = 200

# BioSemi ADC resolution: 1 bit = 31.25 nV = 0.03125 uV
# So 1 uV = 1 / 0.03125 = 32 bits
DEFAULT_UV_TO_BITS = 32
DEFAULT_BYTES_PER_SAMPLE = 3  # 24-bit samples
DEFAULT_SAMPLES_PER_PACKET = 2  # BioSemi sends multiple samples per TCP packet


class DataStreamThread(QThread):
    """Background thread to listen to a TCP socket continuously.
    Decodes BioSemi ActiveView 24-bit LE signed integer format."""
    new_data_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    disconnected_signal = pyqtSignal()

    def __init__(self, host, port, channels, bytes_per_sample, uv_to_bits, samples_per_packet):
        super().__init__()
        self.host = host
        self.port = port
        self.channels = channels
        self.bytes_per_sample = bytes_per_sample
        self.samples_per_packet = samples_per_packet
        self.packet_size = channels * bytes_per_sample * samples_per_packet
        self.running = False
        self.sock = None
        self.bits_to_uv = 0.03125

    def recvall(self, n):
        data = bytearray()
        while len(data) < n and self.running:
            try:
                packet = self.sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except Exception:
                return None
        return data

    def run(self):
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        
        try:
            self.sock.connect((self.host, self.port))
        except Exception as e:
            self.error_signal.emit(f"Failed to connect: {e}")
            self.running = False
            return

        while self.running:
            try:
                raw_data = self.recvall(self.packet_size)
                
                if not self.running:
                    break
                    
                if not raw_data:
                    self.disconnected_signal.emit()
                    self.running = False
                    break
                    
                all_samples = []
                sample_stride = self.channels * self.bytes_per_sample
                for s in range(self.samples_per_packet):
                    sample_offset = s * sample_stride
                    values = []
                    for ch in range(self.channels):
                        offset = sample_offset + ch * self.bytes_per_sample
                        sample_bytes = raw_data[offset: offset + self.bytes_per_sample]
                        raw_int = int.from_bytes(sample_bytes, byteorder='little', signed=True)
                        uv_value = float(raw_int) * self.bits_to_uv
                        values.append(uv_value)
                    all_samples.append(values)
                self.new_data_signal.emit(all_samples)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.error_signal.emit(f"Connection error: {e}")
                self.running = False
                break
                
        self.sock.close()

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
