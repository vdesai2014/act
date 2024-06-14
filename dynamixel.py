import time
import numpy as np
from dynamixel_sdk import *

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_PRESENT_POSITION = 36
LEN_MX_GOAL_POSITION = 4
PROTOCOL_VERSION = 1.0
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

class Dynamixel:
    def __init__(self):
        self.port = 'COM4'
        self.baudrate = 2000000
        self.dynamixel_ids = [0, 1, 2, 3]

        self.portHandler = PortHandler(self.port)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            raise Exception("Failed to open the port")
        print("Port opened successfully")

        if not self.portHandler.setBaudRate(self.baudrate):
            raise Exception("Failed to set baud rate")
        print("Baud rate set successfully")

        for dxl_id in self.dynamixel_ids:
            self._enable_torque(dxl_id)

        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_MX_GOAL_POSITION, LEN_MX_GOAL_POSITION)

        self.reset_pose = [28, 171.2, 186, 38]

    def _enable_torque(self, dxl_id):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
        print(f"Torque enabled for Dynamixel ID {dxl_id}")

    def _disable_torque(self, dxl_id):
        self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
        print(f"Torque disabled for Dynamixel ID {dxl_id}")

    def get_motor_positions(self):
        motor_positions = np.zeros(len(self.dynamixel_ids))
        for i, dxl_id in enumerate(self.dynamixel_ids):
            try:
                dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, dxl_id, ADDR_MX_PRESENT_POSITION)
                motor_positions[i] = (dxl_present_position / 4095) * 360
            except Exception as e:
                motor_positions[i] = 0
        return motor_positions

    def set_motor_positions(self, angles):
        for i, dxl_id in enumerate(self.dynamixel_ids):
            position_value = int(angles[i] / 360 * 4095)
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(position_value)),
                                   DXL_HIBYTE(DXL_LOWORD(position_value)),
                                   DXL_LOBYTE(DXL_HIWORD(position_value)),
                                   DXL_HIBYTE(DXL_HIWORD(position_value))]
            if not self.groupSyncWrite.addParam(dxl_id, param_goal_position):
                print(f"Failed to add param for Dynamixel ID {dxl_id}")
                raise Exception(f"Failed to add param for Dynamixel ID {dxl_id}")

        if self.groupSyncWrite.txPacket() != COMM_SUCCESS:
            raise Exception("Failed to write goal positions")

        self.groupSyncWrite.clearParam()

    def reset(self):
        self.set_motor_positions(self.reset_pose)

    def close(self):
        for dxl_id in self.dynamixel_ids:
            self._disable_torque(dxl_id)
        self.portHandler.closePort()
