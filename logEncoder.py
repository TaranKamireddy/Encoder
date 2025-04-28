import serial
import time

#config
COM_PORT = 'COM3'
BAUD_RATE = 1000000
OUTPUT_FILE = 'arduino_log.csv'

ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

with open(OUTPUT_FILE, 'w', newline='') as f:
    f.write("timestamp,data\n")
    print("Logging started. Press Ctrl+C to stop.\n")

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors='ignore').strip()
                timestamp = time.time()
                f.write(f"{timestamp},{line}\n")
                f.flush()
                print(f"{timestamp}: {line}")
    except KeyboardInterrupt:
        print("\nLogging stopped.")
    finally:                                     
        ser.close()
