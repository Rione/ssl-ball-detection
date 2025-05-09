import serial


class RobotCommunicator:
    def __init__(self, port, baudrate=115200):
        if not port:
            self.serial = None
        else:
            self.port = port
            self.baudrate = baudrate
            self.serial = serial.Serial(port, baudrate)
            print(f"Connected to {port} at {baudrate} baudrate. Opened serial port.")

    def close(self):
        if self.serial is None:
            print("Serial port is not initialized.")
            return

        if self.serial.is_open:
            self.serial.close()
            print(f"Closed serial port {self.port}.")
        else:
            print(f"Serial port {self.port} is already closed.")

    def send_data(self, cameraX: int, cameraY: int):
        if self.serial is None:
            print("Serial port is not initialized.")
            return
        if not self.serial.is_open:
            print(f"Serial port {self.port} is not open.")
            return

        send_byts = [
            0xFF,  # 0. Header
            0x00,  # 1. VelX[LSB]
            0x00,  # 2. VelX[MSB]
            0x00,  # 3. VelY[LSB]
            0x00,  # 4. VelY[MSB]
            0x00,  # 5. VelAngular[LSB]
            0x00,  # 6. VelAngular[MSB]
            0x00,  # 7. dribblePower[0~100] [uint8_t]
            0x00,  # 8. kickerPowerStraight[0~100] [uint8_t]
            0x00,  # 9. kickerPowerChip[0~100] [uint8_t]
            0x00,  # 10. relativePositionX[LSB]
            0x00,  # 11. relativePositionX[MSB]
            0x00,  # 12. relativePositionY[LSB]
            0x00,  # 13. relativePositionY[MSB]
            0x00,  # 14. relativeTheta[LSB]
            0x00,  # 15. relativeTheta[MSB]
            cameraX,  # 16. cameraX [uint8_t]
            cameraY,  # 17. cameraY [uint8_t]
            0b01000000,  # 18. status [uint8_t]
            #  status
            # . bit0: emergencyStop
            # . bit1: doDirectKick
            # . bit2: doDirectChipKick
            # . bit3: reserved
            # . bit4: doCharge
            # . bit5: isSignalReceived
            # . bit6: isCtrlByRobot (0: RACOON-Ctrl, 1: Robot-local-Ctrl)
            # . bit7: parity
        ]

        self.serial.write(bytearray(send_byts))
        # print(f"Sent data: {send_byts}")

    def receive_data(self):
        if self.serial is None:
            print("Serial port is not initialized.")
            return
        if not self.serial.is_open:
            print(f"Serial port {self.port} is not open.")
            return

        if self.serial.in_waiting > 0:
            data = self.serial.read(self.serial.in_waiting)
            # print(f"Received data: {data}")
            return data
        else:
            return None
