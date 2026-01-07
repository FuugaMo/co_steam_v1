import sounddevice as sd

print('Input devices:')
for i, d in enumerate(sd.query_devices()):
    if d.get('max_input_channels', 0) > 0:
        print(f"{i}: {d.get('name', '')}")

