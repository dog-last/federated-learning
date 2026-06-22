import socket
import struct
import pickle
import gzip
import logging

MAGIC_NUMBER = b'SF26'

class TCPCommunicator:
    """
    Handles robust transmission of Python objects/Tensors over TCP sockets.
    Deals with the sticky-packet problem via prefix framing (e.g. 8-byte length prefix).
    """
    def __init__(self, use_compression=False):
        self.use_compression = use_compression

    @staticmethod
    def _recvall(sock, num_bytes):
        """
        Helper function to ensure exact number of bytes are received.
        Resolves TCP stream chunking / sticky packets.
        """
        data = bytearray()
        while len(data) < num_bytes:
            packet = sock.recv(num_bytes - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def send_data(self, sock, message_dict):
        """
        Serializes and sends a dictionary containing the payload.
        Adds an 8-byte prefix representing the payload length in bytes.
        """
        try:
            serialized_data = pickle.dumps(message_dict)
            if self.use_compression:
                # Provide optional GZIP compression for extensions
                serialized_data = gzip.compress(serialized_data)
            
            # Use 'Q' (unsigned long long) for 8-byte prefix
            msg_length = struct.pack('>Q', len(serialized_data))
            sock.sendall(msg_length + MAGIC_NUMBER + serialized_data)
            return True, len(serialized_data)
        except (pickle.PickleError, OSError, ValueError, TypeError) as e:
            logging.error(f"Error sending data: {e}")
            return False, 0

    def recv_data(self, sock):
        message, _ = self.recv_data_with_meta(sock)
        return message

    def _serialize(self, message_dict):
        """Serialize a dictionary to bytes."""
        data = pickle.dumps(message_dict)
        if self.use_compression:
            data = gzip.compress(data)
        return data

    def _deserialize(self, data):
        """Deserialize bytes to a dictionary."""
        if self.use_compression:
            data = gzip.decompress(data)
        return pickle.loads(data)

    def recv_data_with_meta(self, sock):
        """
        Receives data from bounded streams, waits strictly for exact bytes to handle huge models.
        """
        try:
            # Step 1: receive the 8 byte prefix
            raw_msglen = self._recvall(sock, 8)
            if not raw_msglen:
                return None, None

            # Unpack the 8-byte length
            msg_length = struct.unpack('>Q', raw_msglen)[0]
            
            # Step 2: Receive and verify magic number
            raw_magic = self._recvall(sock, 4)
            if raw_magic != MAGIC_NUMBER:
                raise ValueError(f"Magic number mismatch. Expected {MAGIC_NUMBER}, got {raw_magic}")

            # Step 3: receive the actual payload based on strict size
            raw_payload = self._recvall(sock, msg_length)
            if not raw_payload:
                return None, None

            # Decompress if applicable
            if self.use_compression:
                raw_payload = gzip.decompress(raw_payload)

            # Step 4: deserialize pickle
            message_dict = pickle.loads(raw_payload)
            return message_dict, {
                "payload_bytes": int(msg_length),
                "compression": bool(self.use_compression),
                "magic_ok": True,
            }

        except ConnectionResetError:
            logging.warning("Connection was reset by peer.")
            return None, None
        except ValueError as ve:
            logging.error(f"Validation Error: {ve}")
            raise ve
        except (pickle.PickleError, gzip.BadGzipFile, OSError, struct.error, EOFError) as e:
            logging.error(f"Defect during recv: {e}")
            return None, None
